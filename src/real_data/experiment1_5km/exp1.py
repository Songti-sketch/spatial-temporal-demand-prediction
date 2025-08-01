# import pandas as pd
# import numpy as np
# # from preprocessing3 import Preprocessor
# # from estimation3 import Estimator, EstimationResult
# from main3 import TemporalAnalyzer
# import multiprocessing
# import os
# from datetime import datetime, timedelta

# def time_period_in_minutes(start_time, end_time):
#     """
#     Calculate the time period in minutes between two time strings.
    
#     Args:
#         start_time (str): Start time in 'HH:MM' or 'HH:MM:SS' format
#         end_time (str): End time in 'HH:MM' or 'HH:MM:SS' format
    
#     Returns:
#         float: Time period in minutes
    
#     Raises:
#         ValueError: If time strings are in an invalid format
#     """
#     # Try parsing with seconds first
#     try:
#         start_dt = datetime.strptime(start_time, '%H:%M:%S')
#         end_dt = datetime.strptime(end_time, '%H:%M:%S')
#     except ValueError:
#         # If that fails, try without seconds
#         try:
#             start_dt = datetime.strptime(start_time, '%H:%M')
#             end_dt = datetime.strptime(end_time, '%H:%M')
#         except ValueError:
#             raise ValueError("Invalid time format. Use 'HH:MM' or 'HH:MM:SS'.")
    
#     # Calculate the difference
#     delta = end_dt - start_dt
    
#     # If end time is earlier than start time, assume it's on the next day
#     if delta.total_seconds() < 0:
#         delta += timedelta(days=1)
    
#     # Convert to minutes and return
#     return delta.total_seconds() / 60

# def run_experiment1(params):
#     dist_per_grid, time_windows, tw_id = params

#     data_path = '/home/go3/wch_code/jx/real_data/data/cleaned_data2.csv'  # Adjust this path to your data file
#     # df = Preprocessor.clean_transform(data_path)
#     df = pd.read_csv(data_path)
    
#     training_dates = ['2022-10-17', '2022-10-18', '2022-10-19']
#     testing_dates = ['2022-10-20', '2022-10-21']
#     # timeperiod = 240  # Fixed time aggregation level in minutes
#     timeperiods = pd.to_datetime(time_windows[0][1]) - pd.to_datetime(time_windows[0][0])
#     timeperiods = int(timeperiods.total_seconds() / 60)

#     results = TemporalAnalyzer.run_analysis(
#         df=df,
#         time_intervals=time_windows,
#         dist_per_grid=dist_per_grid,
#         # time periods between time_windows
#         timeperiod= timeperiods,
#         training_dates=training_dates,
#         testing_dates=testing_dates,
#         max_iter=100,
#         tol=5e-4,
#         grid_dependent=True,
#         penalty=0
#     )
    
#     save_dir = 'results_5km_3/'
#     os.makedirs(save_dir, exist_ok=True)
#     # save results into .npz file
#     np.savez(
#         os.path.join(save_dir, f'results_dpg_{dist_per_grid}_{tw_id}.npz'),
#         results=results
#     )

#     save_dir = f'plots_5km_3/dpg_{dist_per_grid}_tw_{tw_id}'
#     os.makedirs(save_dir, exist_ok=True)
#     TemporalAnalyzer.visualize_results(results, save_dir)

# dist_per_grid_list = [
#     # 0.1,
#     # 0.2,
#     # 0.3
#     # 0.4,
#     # # 0.5,
#     # 0.6,
#     # 0.7,
#     # 0.8,
#     # 0.9
#     # 0.75
#     # , 
#     # 1.0, 1.5, 
#     2.0, 2.5, 
#     3.0,
#     # , 4.0,
#     # 5.0
#     ]  # List of distance per grid values

# # Task 1: Fix grid size and time period, look at the WMAPE

# # Define different time_windows configurations with identifiers
# time_windows_configs = [
#     ('full_day', [
#         # ('00:00', '01:59:59'), 
#         # ('02:00', '03:59:59'), 
#         # ('04:00', '05:59:59'),
#         ('08:00', '09:59:59'), 
#         ('10:00', '11:59:59'), 
#         ('12:00', '13:59:59'),
#         ('14:00', '15:59:59'), 
#         ('16:00', '17:59:59'), 
#         ('18:00', '19:59:59'),
#         ('20:00', '21:59:59'), 
#         ('22:00', '23:59:59')
#     ])
#     # ,
#     # # ('morning_afternoon', [
#     # #     ('06:00', '12:00'), 
#     # #     ('12:00', '18:00')
#     # # ]),
#     # ('peak_hours_lunch', [
#     #     ('10:00', '14:00')
#     #     # ('16:00', '20:00')
#     # ]),
#     # ('peak_hours_lunch', [
#     #     # ('10:00', '14:00')
#     #     ('16:00', '20:00')
#     # ])
# ]

# # Create experiment combinations
# experiments = [
#     (dpg, tw, tw_id)
#     for tw_id, tw in time_windows_configs
#     for dpg in dist_per_grid_list
# ]

# # Run experiments in parallel
# # max number of processors
# max_processors = (multiprocessing.cpu_count() - 1)//2
# with multiprocessing.Pool(processes=max_processors) as pool:
#     pool.map(run_experiment1, experiments)


import pandas as pd
import numpy as np
from main3_0623 import TemporalAnalyzer
import multiprocessing
import os
from datetime import datetime, timedelta
from itertools import product

def time_period_in_minutes(start_time, end_time):
    """
    Calculate the time period in minutes between two time strings.
    
    Args:
        start_time (str): Start time in 'HH:MM' or 'HH:MM:SS' format
        end_time (str): End time in 'HH:MM' or 'HH:MM:SS' format
    
    Returns:
        float: Time period in minutes
    
    Raises:
        ValueError: If time strings are in an invalid format
    """
    try:
        start_dt = datetime.strptime(start_time, '%H:%M:%S')
        end_dt = datetime.strptime(end_time, '%H:%M:%S')
    except ValueError:
        try:
            start_dt = datetime.strptime(start_time, '%H:%M')
            end_dt = datetime.strptime(end_time, '%H:%M')
        except ValueError:
            raise ValueError("Invalid time format. Use 'HH:MM' or 'HH:MM:SS'.")
    
    delta = end_dt - start_dt
    if delta.total_seconds() < 0:
        delta += timedelta(days=1)
    return delta.total_seconds() / 60

def run_single_combination(args):
    """Run analysis for a single combination of dist_per_grid and time window."""
    dpg, tw, df, timeperiod, training_dates, testing_dates = args
    result = TemporalAnalyzer.run_analysis(
        df=df,
        time_intervals=[tw],
        dist_per_grid=dpg,
        timeperiod=timeperiod,
        training_dates=training_dates,
        testing_dates=testing_dates,
        max_iter=100,
        tol=5e-4,
        penalty=0
    )
    print(f"Completed analysis for dist_per_grid={dpg}, time_window={tw}")
    return dpg, tw, result

def process_all_combinations():
    """Process all combinations of dist_per_grid and time windows in parallel."""
    # Data and configuration
    data_path = '/home/go3/wch_code/jx/real_data/data/cleaned_data2.csv'
    df = pd.read_csv(data_path)
    training_dates = ['2022-10-17', '2022-10-18', '2022-10-19']
    testing_dates = ['2022-10-20', '2022-10-21']
    
    # Define distance and time window configurations
    dist_per_grid_list = [
        # 0.3, 0.4,
        0.5,
        # 0.6,
        # 0.7, 0.8, 0.9,
        1.0, 1.5, 
        2.0, 2.5, 
        3.0, 
        4.0,
        5.0
        ]
    time_windows_configs = [
        ('full_day', [
            ('08:00', '10:00'),
            ('10:00', '12:00'),
            ('12:00', '14:00'),
            ('14:00', '16:00'),
            ('16:00', '18:00'),
            ('18:00', '20:00'),
            ('20:00', '22:00'),
            ('22:00', '23:59')
            # ('08:00:00', '08:59:59'), 
            # ('09:00:00', '09:59:59'), 
            # ('10:00:00', '10:59:59'), 
            # ('11:00:00', '11:59:59'), 
            # ('12:00:00', '12:59:59'),
            # ('13:00:00', '13:59:59'),
            # ('14:00:00', '14:59:59'), 
            # ('15:00:00', '15:59:59'), 
            # ('16:00:00', '16:59:59'),
            # ('17:00:00', '17:59:59'), 
            # ('18:00:00', '18:59:59'),
            # ('19:00:00', '19:59:59'),
            # ('20:00:00', '20:59:59'),
            # ('21:00:00', '21:59:59'), 
            # ('22:00:00', '22:59:59'),
            # ('23:00:00', '23:59:59')
        ])
    ]
    
    time_windows = time_windows_configs[0][1]  # Use 'full_day' time windows
    timeperiod = time_period_in_minutes(time_windows[0][0], time_windows[0][1])
    # max_processors = (multiprocessing.cpu_count() - 1)
    max_processors = 64  # Ensure at least one processor is used

    # Generate all combinations of dist_per_grid and time windows
    combinations = list(product(dist_per_grid_list, time_windows))

    # Run analysis for each combination in parallel
    with multiprocessing.Pool(processes=max_processors) as pool:
        args_list = [(dpg, tw, df, timeperiod, training_dates, testing_dates) 
                    for dpg, tw in combinations]
        results_list = pool.map(run_single_combination, args_list)

    # Organize results into a dictionary
    results_dict = {(dpg, tw): res for dpg, tw, res in results_list}

    # Save results
    save_dir = 'results_5km_3/'
    os.makedirs(save_dir, exist_ok=True)
    for dpg in dist_per_grid_list:
        dpg_results = {f"tw_{tw[0]}_{tw[1]}": results_dict[(dpg, tw)] for tw in time_windows}
        np.savez(
            os.path.join(save_dir, f'results_dpg_{dpg}_full_day.npz'),
            results=dpg_results
        )

    # Visualize results
    for dpg in dist_per_grid_list:
        visualize_dir = f'plots_5km_3/dpg_{dpg}_tw_full_day'
        os.makedirs(visualize_dir, exist_ok=True)
        dpg_results = {f"tw_{tw[0]}_{tw[1]}": results_dict[(dpg, tw)] for tw in time_windows}
        TemporalAnalyzer.visualize_results(dpg_results, visualize_dir)