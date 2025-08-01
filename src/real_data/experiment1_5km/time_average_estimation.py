import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
from datetime import datetime, timedelta

# Assuming Preprocessor and EstimationResult are defined elsewhere
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
        self.p_ij_avg = None
        self.o_ij_avg = None
        self.in_relative_errors = None
        self.out_relative_errors = None
        self.in_o_ij_actual = None
        self.out_o_ij_actual = None

def run_time_average_method(params):
    dist_per_grid, time_windows, tw_id = params

    # Load and preprocess data
    data_path = '/home/go3/wch_code/jx/real_data/data/cleaned_data2.csv'
    df = pd.read_csv(data_path)
    
    training_dates = ['2022-10-17', '2022-10-18', '2022-10-19']
    testing_dates = ['2022-10-20', '2022-10-21']

    training_dates = [pd.to_datetime(date).date() for date in training_dates]
    testing_dates = [pd.to_datetime(date).date() for date in testing_dates]
    all_dates = training_dates + testing_dates
    df = Preprocessor.preprocess(df, all_dates)

    timeperiods = pd.to_datetime(time_windows[0][1]) - pd.to_datetime(time_windows[0][0])
    timeperiods = int(timeperiods.total_seconds() / 60)
    
    base_df, num_total_cells, _, _, _, _, _ = Preprocessor.cut_df(df, dist_per_grid, timeperiods)

    results = []
    for win_start, win_end in time_windows:
        print(f"Processing time window: {win_start} to {win_end}")

        in_relative_errors = []
        out_relative_errors = []
        in_o_ij_actuals = {}
        out_o_ij_actuals = {}
        o_ij_avgs = []
        p_avgs = []

        start_time = pd.to_datetime(win_start).time()
        end_time = pd.to_datetime(win_end).time()
        
        # Filter training data for the current time window
        win_df_train = base_df[(base_df['date'].isin(training_dates)) & 
                               (base_df['time'] >= start_time) & 
                               (base_df['time'] <= end_time)].copy()
        win_df_train['time_window'] = f"{win_start}-{win_end}"

        o_ij_avg = np.zeros((num_total_cells, num_total_cells))
        p_avg = np.zeros((num_total_cells, num_total_cells))
        if len(win_df_train) > 0:
            # Calculate average o_ij over training data
            customers = win_df_train['cust_index'].unique()
            sellers = win_df_train['sell_index'].unique()

            for cust in customers:
                for sell in sellers:
                    num_choices = len(win_df_train[(win_df_train['cust_index'] == cust) & 
                                                  (win_df_train['sell_index'] == sell)])
                    o_ij_avg[cust, sell] = num_choices / 3
                # Normalize o_ij_avg to sum to 1 for each customer
                p_avg[cust, :] = o_ij_avg[cust, :] / (o_ij_avg[cust, :].sum()) if o_ij_avg[cust, :].sum() > 0 else 0

            o_ij_avgs.append(o_ij_avg)
            p_avgs.append(p_avg)

            # Evaluate on test data and calculate WMAPE
            test_set = ['in', 'out']
            
            for i in test_set:
                if i == 'in':
                    dates = training_dates
                if i == 'out':
                    dates = testing_dates

                for date in dates:
                    win_df = base_df[(base_df['date'] == date) & 
                                     (base_df['time'] >= start_time) & 
                                     (base_df['time'] <= end_time)].copy()
                    if len(win_df) > 0:
                        o_ij = np.zeros((num_total_cells, num_total_cells))
                        p_ij = np.zeros((num_total_cells, num_total_cells))
                        customers = win_df['cust_index'].unique()
                        sellers = win_df['sell_index'].unique()
                        for cust in customers:
                            for sell in sellers:
                                num_choices = len(win_df[(win_df['cust_index'] == cust) & 
                                                        (win_df['sell_index'] == sell)])
                                o_ij[cust, sell] = num_choices
                            p_ij[cust, :] = o_ij[cust, :] / (o_ij[cust, :].sum()) if o_ij[cust, :].sum() > 0 else 0
                        
                        # in_o_ij_actuals[date] = o_ij if i == 'in' else out_o_ij_actuals[date] = o_ij
                        
                        if i == 'in':
                            in_o_ij_actuals[date] = o_ij
                        else:
                            out_o_ij_actuals[date] = o_ij
                            
                        p_ij = p_ij.flatten()
                        p_avg = p_avg.flatten()
                        # error = error.flatten()
                        error = np.abs(p_ij - p_avg)
                        # error = error[~np.isinf(error) & ~np.isnan(error)]
                        
                        print(f"Relative error for {i} on date {date}: {error}")
                        
                        
                        in_relative_errors.append(error) if i == 'in' else out_relative_errors.append(error)
                        
            # in_reletive_errors = []
            # for train_date in training_dates:
            #     win_df_train = base_df[(base_df['date'] == train_date) & 
            #                           (base_df['time'] >= start_time) & 
            #                           (base_df['time'] <= end_time)].copy()
            #     if len(win_df_train) > 0:
            #         o_ij = np.zeros((num_total_cells, num_total_cells))
            #         customers = win_df_train['cust_index'].unique()
            #         sellers = win_df_train['sell_index'].unique()
            #         for cust in customers:
            #             for sell in sellers:
            #                 num_choices = len(win_df_train[(win_df_train['cust_index'] == cust) & 
            #                                             (win_df_train['sell_index'] == sell)])
            #                 o_ij[cust, sell] = num_choices
            #             # Normalize o_ij to sum to 1 for each customer
            #             o_ij[cust, :] = o_ij[cust, :] / (o_ij[cust, :].sum()) if o_ij[cust, :].sum() > 0 else 0
            #         # o_ij = o_ij / (o_ij.sum(axis=1, keepdims=True))

            #         o_ij = o_ij.flatten()
            #         o_ij_avg = o_ij_avg.flatten()
            #         # error = error.flatten()
            #         error = (np.abs(o_ij - o_ij_avg) / (o_ij)) * 100 
            #         error = error[~np.isinf(error) & ~np.isnan(error)]
            #         in_reletive_errors.append(error)

            # out_reletive_errors = []
            # for test_date in testing_dates:
            #     win_df_test = base_df[(base_df['date'] == test_date) & 
            #                           (base_df['time'] >= start_time) & 
            #                           (base_df['time'] <= end_time)].copy()
            #     if len(win_df_test) > 0:
            #         o_ij = np.zeros((num_total_cells, num_total_cells))
            #         customers = win_df_test['cust_index'].unique()
            #         sellers = win_df_test['sell_index'].unique()
            #         for cust in customers:
            #             for sell in sellers:
            #                 num_choices = len(win_df_test[(win_df_test['cust_index'] == cust) & 
            #                                             (win_df_test['sell_index'] == sell)])
            #                 o_ij[cust, sell] = num_choices
            #         o_ij = o_ij / (o_ij.sum(axis=1, keepdims=True) + 1e-6) if o_ij.sum() > 0 else o_ij

            #         error = (np.abs(o_ij - o_ij_avg) / (o_ij + 1e-6)) * 100 
            #         error = error.flatten()
            #         error = error[~np.isinf(error) & ~np.isnan(error)]
            #         out_reletive_errors.append(error)

            # Store results
            result = EstimationResult(f"{win_start}-{win_end}")
            result.p_ij_avg = p_avgs
            result.o_ij_avg = o_ij_avgs
            result.in_relative_errors = in_relative_errors
            result.out_relative_errors = out_relative_errors
            result.in_o_ij_actual = in_o_ij_actuals
            result.out_o_ij_actual = out_o_ij_actuals

            results.append(result)
    
    # Save results
    save_dir = 'time_average/results'
    os.makedirs(save_dir, exist_ok=True)
    np.savez(
        os.path.join(save_dir, f'results_dpg_{dist_per_grid}_{tw_id}.npz'),
        results=results
    )

    # Visualize results
    save_dir = 'time_average/plots'
    # os.makedirs(save_dir, exist_ok=True)
    # if save_dir not exist, create it
    if not os.path.exists(f'time_average/plots/dpg_{dist_per_grid}_tw_{tw_id}'):
        os.makedirs(f'time_average/plots/dpg_{dist_per_grid}_tw_{tw_id}')
    # visualize_time_average_results(results, f'time_average/plots/dpg_{dist_per_grid}_tw_{tw_id}')

def visualize_time_average_results(results, save_dir):
    # Box plot for relative errors
    font_size = 15
    
    # Prepare data for in-sample relative errors, filtering above 95th quantile
    in_relative_errors_dict = {}
    in_mean = {}
    for r in results:
        if r.in_relative_errors:  # Check if there are any errors
            errors = np.concatenate(r.in_relative_errors)  # Flatten list of arrays
            if len(errors) > 0:  # Ensure there are values to process
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
    
    # Create boxplot for in-sample relative errors
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=in_relative_errors_df)
    
    # Add mean markers
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
    plt.show()
    
    # Prepare data for out-of-sample relative errors, filtering above 95th quantile
    out_relative_errors_dict = {}
    out_mean = {}
    for r in results:
        if r.out_relative_errors:  # Check if there are any errors
            errors = np.concatenate(r.out_relative_errors)  # Flatten list of arrays
            if len(errors) > 0:  # Ensure there are values to process
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
    
    # Create boxplot for out-of-sample relative errors
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=out_relative_errors_df)
    
    # Add mean markers
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
    plt.show()

if __name__ == "__main__":
    max_processors = (multiprocessing.cpu_count() - 1)//2
    dist_per_grid_list = [
        # 0.1,
        # 0.2,
        # 0.3,
        # 0.4, 
        # 0.6, 0.7
        # , 0.8, 0.9
        0.5
        # , 0.75
        , 1.0, 1.5, 2.0, 2.5, 3.0
        # 5.0
        ]
    time_windows_configs = [
        ('full_day', [
            ('08:00', '09:59:59'), 
            ('10:00', '11:59:59'), 
            ('12:00', '13:59:59'),
            ('14:00', '15:59:59'), 
            ('16:00', '17:59:59'), 
            ('18:00', '19:59:59'),
            ('20:00', '21:59:59'), 
            ('22:00', '23:59:59')
        ])
    ]
    experiments = [
        (dpg, tw, tw_id)
        for tw_id, tw in time_windows_configs
        for dpg in dist_per_grid_list
    ]
    with multiprocessing.Pool(processes=max_processors) as pool:
        pool.map(run_time_average_method, experiments)