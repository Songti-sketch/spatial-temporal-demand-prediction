import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pdb

class Preprocessor:
    @staticmethod
    def unix_to_UTC8(unix_timestamp):
        if unix_timestamp == 0:
            return None
        time = datetime(1970, 1, 1) + timedelta(seconds=int(unix_timestamp) + 8 * 3600)
        return pd.to_datetime(time)
    
    def clean_tranform(data_path='to/your/data.csv'):
        df = pd.read_csv(data_path)
        unix_columns_orders = [
            'platform_order_time', 'estimate_arrived_time', 'estimate_meal_prepare_time',
            'order_push_time', 'dispatch_time', 'grab_time', 'fetch_time', 'arrive_time'
        ]
        for i in unix_columns_orders:
            df[i] = np.vectorize(Preprocessor.unix_to_UTC8)(df[i])
        df['sender_lat'] = df['sender_lat'] / 1e6
        df['sender_lng'] = df['sender_lng'] / 1e6
        df['recipient_lat'] = df['recipient_lat'] / 1e6
        df['recipient_lng'] = df['recipient_lng'] / 1e6
        df['estimate_duration'] = df['estimate_arrived_time'] - df['platform_order_time']
        df = df[df['estimate_duration'].notnull()]
        df = df[df['is_courier_grabbed'] == 1]
        return df
    
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
    
    def get_latlng_from_id(index, num_cells_lon):
        i = index // num_cells_lon
        j = index % num_cells_lon
        return i, j

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

    @staticmethod
    def split_time_windows(df, time_intervals):
        dfs = {}
        df['time'] = pd.to_datetime(df['platform_order_time']).dt.time
        for i, (start, end) in enumerate(time_intervals):
            start_time = pd.to_datetime(start).time()
            end_time = pd.to_datetime(end).time()
            df['time_window'] = f"{start}-{end}"
            window_df = df[(df['time'] >= start_time) & (df['time'] <= end_time)].copy()
            dfs[f"{start}-{end}"] = window_df
        return dfs

    @staticmethod
    def compute_fixed_assortments(df, dist, num_total_cells, dist_threshold=5.0):

        fixed_assortments = np.zeros((num_total_cells, num_total_cells))
        for i in range(num_total_cells):
            grid_df = df[df['cust_index'] == i]
            chosen_sellers = grid_df['sell_index'].unique()
            fixed_assortments[i, chosen_sellers] = 1
        # mask = (dist <= dist_threshold).astype(float)
        # fixed_assortments = fixed_assortments * mask # shape (num_total_cells, num_total_cells)
        # fixed_assortments = mask
        return fixed_assortments