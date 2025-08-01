import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datetime import datetime, timedelta
from itertools import product
from collections import defaultdict

# Assuming Preprocessor and EstimationResult classes are defined as in the original code
class Preprocessor:
    @staticmethod
    def unix_to_UTC8(unix_timestamp):
        if unix_timestamp == 0:
            return None
        time = datetime(1970, 1, 1) + timedelta(seconds=int(unix_timestamp) + 8*3600)
        return pd.to_datetime(time)

    @staticmethod
    def preprocess(df, dates, start_date='2022-10-17 00:00:00', end_date='2022-10-25 00:00:00'):
        df = df.loc[(df['platform_order_time'] >= start_date) & (df['platform_order_time'] <= end_date)].copy()
        df['platform_order_time'] = pd.to_datetime(df['platform_order_time'])
        df['time'] = df['platform_order_time'].dt.time
        df['date'] = df['platform_order_time'].dt.date
        dates = [pd.to_datetime(date).date() for date in dates]
        df = df.loc[df['date'].isin(dates)]
        return df

    @staticmethod
    def retg_frame(df, lng1, lng2, lat1, lat2):
        df = df[
            (df['sender_lng'] >= lng1) & (df['sender_lng'] <= lng2) &
            (df['sender_lat'] >= lat1) & (df['sender_lat'] <= lat2) &
            (df['recipient_lng'] >= lng1) & (df['recipient_lng'] <= lng2) &
            (df['recipient_lat'] >= lat1) & (df['recipient_lat'] <= lat2)
        ]
        return df
    
    @staticmethod
    def get_grid_id(lon, lat, lon_min, lat_min, lon_step, lat_step, num_cells_lon, num_cells_lat):
        i = int(np.floor((lat - lat_min) / lat_step))
        j = int(np.floor((lon - lon_min) / lon_step))
        i = min(max(i, 0), num_cells_lat - 1)
        j = min(max(j, 0), num_cells_lon - 1)
        index = i * num_cells_lon + j
        return i, j, index
    
    @staticmethod
    def cut_df(df, dist_per_grid=0.5, timeperiod=5):
        df = df[['dt', 'order_id', 'poi_id', 'sender_lng', 'sender_lat',
                 'recipient_lng', 'recipient_lat', 'platform_order_time', 'time']]
        df = Preprocessor.retg_frame(df, 174.447426, 174.653673, 45.800414, 45.945237)
        lon_min, lon_max = 174.447426, 174.653673
        lat_min, lat_max = 45.800414, 45.945237
        KM_PER_DEG_LAT = 111.1949266
        KM_PER_DEG_LON = 77.52014075
        lat_step = dist_per_grid / KM_PER_DEG_LAT
        lon_step = dist_per_grid / KM_PER_DEG_LON
        num_cells_lon = int(np.ceil((lon_max - lon_min) / lon_step))
        num_cells_lat = int(np.ceil((lat_max - lat_min) / lat_step))
        num_total_cells = num_cells_lon * num_cells_lat
        df[['sell_grid_lat', 'sell_grid_lng', 'sell_index']] = df.apply(
            lambda row: pd.Series(Preprocessor.get_grid_id(
                row['sender_lng'], row['sender_lat'], lon_min, lat_min,
                lon_step, lat_step, num_cells_lon, num_cells_lat
            )), axis=1
        )
        df[['cust_grid_lat', 'cust_grid_lng', 'cust_index']] = df.apply(
            lambda row: pd.Series(Preprocessor.get_grid_id(
                row['recipient_lng'], row['recipient_lat'], lon_min, lat_min,
                lon_step, lat_step, num_cells_lon, num_cells_lat
            )), axis=1
        )
        df['platform_order_time'] = pd.to_datetime(df['platform_order_time'])
        df['date'] = df['platform_order_time'].dt.date
        df['timeperiod'] = df['platform_order_time'].dt.floor(f"{timeperiod}min")
        df['time_index'] = df.groupby('date')['timeperiod'].rank(method='dense').astype(int)
        return df, num_total_cells, num_cells_lon, lon_min, lat_min, lon_step, lat_step

class EstimationResult:
    def __init__(self, time_window):
        self.time_window = time_window
        self.o_ij_avg = None
        self.in_relative_errors = None
        self.out_relative_errors = None

#### Worker Function for a Single dpg and Time Window
def run_single_time_window(args):
    """Process a single combination of dist_per_grid and time window."""
    dpg, tw, df, training_dates, testing_dates = args
    win_start, win_end = tw
    
    # Calculate time period in minutes for preprocessing
    timeperiods = (pd.to_datetime(win_end) - pd.to_datetime(win_start)).total_seconds() / 60
    base_df, num_total_cells, _, _, _, _, _ = Preprocessor.cut_df(df, dpg, timeperiods)
    
    start_time = pd.to_datetime(win_start).time()
    end_time = pd.to_datetime(win_end).time()
    
    # Filter training data for the current time window
    win_df_train = base_df[(base_df['date'].isin(training_dates)) & 
                           (base_df['time'] >= start_time) & 
                           (base_df['time'] <= end_time)].copy()
    win_df_train['time_window'] = f"{win_start}-{win_end}"

    # Compute average o_ij matrix
    o_ij_avg = np.zeros((num_total_cells, num_total_cells))
    if len(win_df_train) > 0:
        customers = win_df_train['cust_index'].unique()
        sellers = win_df_train['sell_index'].unique()
        for cust in customers:
            for sell in sellers:
                num_choices = len(win_df_train[(win_df_train['cust_index'] == cust) & 
                                              (win_df_train['sell_index'] == sell)])
                o_ij_avg[cust, sell] = num_choices / 3
            o_ij_avg[cust, :] = o_ij_avg[cust, :] / o_ij_avg[cust, :].sum() if o_ij_avg[cust, :].sum() > 0 else 0

    # Evaluate on in-sample and out-of-sample data
    in_relative_errors = []
    out_relative_errors = []
    for i in ['in', 'out']:
        dates = training_dates if i == 'in' else testing_dates
        for date in dates:
            win_df = base_df[(base_df['date'] == date) & 
                             (base_df['time'] >= start_time) & 
                             (base_df['time'] <= end_time)].copy()
            if len(win_df) > 0:
                o_ij = np.zeros((num_total_cells, num_total_cells))
                customers = win_df['cust_index'].unique()
                sellers = win_df['sell_index'].unique()
                for cust in customers:
                    for sell in sellers:
                        num_choices = len(win_df[(win_df['cust_index'] == cust) & 
                                                (win_df['sell_index'] == sell)])
                        o_ij[cust, sell] = num_choices
                    o_ij[cust, :] = o_ij[cust, :] / o_ij[cust, :].sum() if o_ij[cust, :].sum() > 0 else 0

                o_ij_flat = o_ij.flatten()
                o_ij_avg_flat = o_ij_avg.flatten()
                # Compute relative error only for non-zero o_ij values
                mask = o_ij_flat > 0
                if mask.any():
                    error = (np.abs(o_ij_flat[mask] - o_ij_avg_flat[mask]) / o_ij_flat[mask]) * 100
                    if i == 'in':
                        in_relative_errors.append(error)
                    else:
                        out_relative_errors.append(error)

    # Store result
    result = EstimationResult(f"{win_start}-{win_end}")
    result.o_ij_avg = o_ij_avg
    result.in_relative_errors = in_relative_errors
    result.out_relative_errors = out_relative_errors
    return dpg, tw, result

#### Visualization Function (Unchanged)
def visualize_time_average_results(results, save_dir):
    font_size = 15
    
    # In-sample relative errors
    in_relative_errors_dict = {}
    in_mean = {}
    for r in results:
        if r.in_relative_errors:
            errors = np.concatenate(r.in_relative_errors) if r.in_relative_errors else np.array([])
            if len(errors) > 0:
                quantile_95 = np.quantile(errors, 0.95)
                filtered_errors = errors[errors <= quantile_95]
                in_relative_errors_dict[r.time_window] = filtered_errors
                in_mean[r.time_window] = np.mean(filtered_errors)
            else:
                in_relative_errors_dict[r.time_window] = []
                in_mean[r.time_window] = np.nan
        else:
            in_relative_errors_dict[r.time_window] = []
            in_mean[r.time_window] = np.nan
    
    in_relative_errors_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in in_relative_errors_dict.items()]))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=in_relative_errors_df)
    for idx, time_window in enumerate(in_relative_errors_df.columns):
        mean_val = in_mean[time_window]
        if not np.isnan(mean_val):
            plt.plot(idx, mean_val, 'rD', markersize=8, label='Mean' if idx == 0 else '')
    
    plt.title('In-sample Relative Errors Across Time Windows (Below 95th Quantile)', fontsize=font_size)
    plt.ylabel('Error (%)', fontsize=font_size)
    plt.xlabel('Time Window', fontsize=font_size)
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'in_relative_errors.png'))
    plt.close()
    
    # Out-of-sample relative errors
    out_relative_errors_dict = {}
    out_mean = {}
    for r in results:
        if r.out_relative_errors:
            errors = np.concatenate(r.out_relative_errors) if r.out_relative_errors else np.array([])
            if len(errors) > 0:
                quantile_95 = np.quantile(errors, 0.95)
                filtered_errors = errors[errors <= quantile_95]
                out_relative_errors_dict[r.time_window] = filtered_errors
                out_mean[r.time_window] = np.mean(filtered_errors)
            else:
                out_relative_errors_dict[r.time_window] = []
                out_mean[r.time_window] = np.nan
        else:
            out_relative_errors_dict[r.time_window] = []
            out_mean[r.time_window] = np.nan
    
    out_relative_errors_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in out_relative_errors_dict.items()]))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=out_relative_errors_df)
    for idx, time_window in enumerate(out_relative_errors_df.columns):
        mean_val = out_mean[time_window]
        if not np.isnan(mean_val):
            plt.plot(idx, mean_val, 'rD', markersize=8, label='Mean' if idx == 0 else '')
    
    plt.title('Out-of-sample Relative Errors Across Time Windows (Below 95th Quantile)', fontsize=font_size)
    plt.ylabel('Error (%)', fontsize=font_size)
    plt.xlabel('Time Window', fontsize=font_size)
    plt.xticks(rotation=45)
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'out_relative_errors.png'))
    plt.close()

#### Main Execution
if __name__ == "__main__":
    # Define parameters
    max_processors = (multiprocessing.cpu_count() - 1)
    dist_per_grid_list = [
        0.3, 0.4,
        0.5,
        0.6, 
        # 0.7, 0.8, 0.9,
        # 1.0, 
        1.5, 
        2.0, 2.5, 
        3.0, 
        4.0, 5.0
        ]
    time_windows = [
        ('08:00', '08:59:59'), ('09:00', '09:59:59'), ('10:00', '10:59:59'),
        ('11:00', '11:59:59'), ('12:00', '12:59:59'), ('13:00', '13:59:59'),
        ('14:00', '14:59:59'), ('15:00', '15:59:59'), ('16:00', '16:59:59'),
        ('17:00', '17:59:59'), ('18:00', '18:59:59'), ('19:00', '19:59:59'),
        ('20:00', '20:59:59'), ('21:00', '21:59:59'), ('22:00', '22:59:59'),
        ('23:00', '23:59:59')
    ]
    training_dates = ['2022-10-17', '2022-10-18', '2022-10-19']
    testing_dates = ['2022-10-20', '2022-10-21']

    # Load and preprocess data once
    data_path = '/home/go3/wch_code/jx/real_data/data/cleaned_data2.csv'
    df = pd.read_csv(data_path)
    training_dates = [pd.to_datetime(date).date() for date in training_dates]
    testing_dates = [pd.to_datetime(date).date() for date in testing_dates]
    all_dates = training_dates + testing_dates
    df = Preprocessor.preprocess(df, all_dates)

    # Create experiments: all combinations of dpg and individual time windows
    experiments = list(product(dist_per_grid_list, time_windows))
    args_list = [(dpg, tw, df, training_dates, testing_dates) for dpg, tw in experiments]

    # Parallel processing
    with multiprocessing.Pool(processes=max_processors) as pool:
        results_list = pool.map(run_single_time_window, args_list)

    # Group results by dpg
    results_by_dpg = defaultdict(list)
    for dpg, tw, result in results_list:
        results_by_dpg[dpg].append(result)

    # Sort results by time window start time and process
    save_dir = 'time_average_1h_0623'
    results_dir = os.path.join(save_dir, 'results')
    plots_dir = os.path.join(save_dir, 'plots')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    for dpg in results_by_dpg:
        # Sort by start time of time window
        results_by_dpg[dpg].sort(key=lambda r: pd.to_datetime(r.time_window.split('-')[0]))
        results = results_by_dpg[dpg]
        
        # Save results
        np.savez(
            os.path.join(results_dir, f'results_dpg_{dpg}.npz'),
            results=results
        )
        
        # Visualize
        visualize_dir = os.path.join(plots_dir, f'dpg_{dpg}')
        os.makedirs(visualize_dir, exist_ok=True)
        visualize_time_average_results(results, visualize_dir)