import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing3 import Preprocessor
from estimation3 import Estimator, EstimationResult
import os
import pdb

class TemporalAnalyzer:
    @staticmethod
    def run_analysis(df, time_intervals, dist_per_grid=0.5, timeperiod=30, training_dates=None, testing_dates=None,
                     max_iter=200, tol=1e-4, penalty=0):
        training_dates = [pd.to_datetime(date).date() for date in training_dates]
        testing_dates = [pd.to_datetime(date).date() for date in testing_dates]
        all_dates = training_dates + testing_dates
        base_df = Preprocessor.preprocess(df, all_dates)


        results = []
        for win_start, win_end in time_intervals:
            print('now compute fixed assortments, for dist_per_grid:', dist_per_grid, 'time_interval:', win_start, '-', win_end)
            # calculate number of cells
            base_df_tw, num_total_cells, num_cells_lon, _, _, _, _ = Preprocessor.cut_df(base_df, dist_per_grid, timeperiod)
            # get assortment matrix
            # fixed_assortments = Preprocessor.compute_fixed_assortments(base_df_tw, dist_per_grid, num_total_cells)

            print(f"Processing time window: {win_start} to {win_end}")
            start_time = pd.to_datetime(win_start).time()
            end_time = pd.to_datetime(win_end).time()
            win_df_train = base_df_tw[(base_df_tw['date'].isin(training_dates)) & 
                                   (base_df_tw['time'] >= start_time) & 
                                   (base_df_tw['time'] <= end_time)].copy()
            win_df_train['time_window'] = f"{win_start}-{win_end}"

            train_fixed_assortments = Preprocessor.compute_fixed_assortments(win_df_train, dist_per_grid, num_total_cells)
            
            if len(win_df_train) > 0:
                omega, beta, lambda_hat, dist, p_m = Estimator.estimate_parameters(num_total_cells, num_cells_lon,
                    win_df_train, dist_per_grid, timeperiod, max_iter=max_iter, tol=tol, fixed_assortments = train_fixed_assortments, penalty=penalty)
                
                in_sample_wmapes = {}
                in_o_ij_actual = {}
                in_o_ij_hat = {}
                in_dist = {}
                in_N_hat = {}
                in_N_actual = {}
                in_sample_relative_errors = []
                for train_date in training_dates:
                    win_df_train = base_df_tw[(base_df_tw['date'] == train_date) & 
                                           (base_df_tw['time'] >= start_time) & 
                                           (base_df_tw['time'] <= end_time)].copy()
                    
                    # train_assort = Preprocessor.compute_fixed_assortments(win_df_train, dist_per_grid, num_total_cells)
                    
                    
                    if len(win_df_train) > 0:
                        win_df_train['time_window'] = f"{win_start}-{win_end}"
                        extra_assortment = train_fixed_assortments
                        in_sample_wmape, relative_errors, o_ij_actual, o_ij_hat, dist, N_hat, N_actual = Estimator.calculate_wmape(
                            extra_assortment,
                            p_m, omega, beta, lambda_hat, dist, 
                            win_df_train, dist_per_grid, timeperiod
                            )
                        in_sample_wmapes[train_date] = in_sample_wmape
                        in_o_ij_actual[train_date] = o_ij_actual
                        in_o_ij_hat[train_date] = o_ij_hat
                        in_dist[train_date] = dist
                        in_N_hat[train_date] = N_hat
                        in_N_actual[train_date] = N_actual
                        in_sample_relative_errors.extend(relative_errors)


                win_df_test = base_df_tw[(base_df_tw['date'].isin(testing_dates)) & 
                                    (base_df_tw['time'] >= start_time) & 
                                    (base_df_tw['time'] <= end_time)].copy()
                win_df_test['time_window'] = f"{win_start}-{win_end}"

                
                # calculate the ratio of overlap 

                out_sample_wmapes = {}
                out_o_ij_actual = {}
                out_o_ij_hat = {}
                out_dist = {}
                out_N_hat = {}
                out_N_actual = {}
                out_sample_relative_errors = []
                out_sample_assort_ratio = {}
                for test_date in testing_dates:
                    win_df_test = base_df_tw[(base_df_tw['date'] == test_date) & 
                                          (base_df_tw['time'] >= start_time) & 
                                          (base_df_tw['time'] <= end_time)].copy()
                    test_fixed_assortments = Preprocessor.compute_fixed_assortments(win_df_test, dist_per_grid, num_total_cells)
                    if len(win_df_test) > 0:
                        out_sample_wmape, relative_errors, o_ij_actual, o_ij_hat, dist, N_hat, N_actual = Estimator.calculate_wmape(
                            test_fixed_assortments,
                            p_m, omega, beta, lambda_hat, dist,
                            win_df_test, dist_per_grid, timeperiod)
                        out_sample_wmapes[test_date] = out_sample_wmape
                        out_o_ij_actual[test_date] = o_ij_actual
                        out_o_ij_hat[test_date] = o_ij_hat
                        out_dist[test_date] = dist
                        out_N_hat[test_date] = N_hat
                        out_N_actual[test_date] = N_actual
                        out_sample_relative_errors.extend(relative_errors)

                        # calculate the ratio of overlap between the test assortment and the training assortment
                        # first use train_fixed_assortments - test_fixed_assortments, then calculate number of 0 in each row of train_fixed_assortments, divided by number of 1 in each row of train_fixed_assortments
                        hit = train_fixed_assortments * test_fixed_assortments
                        # count the number of 0 in each row of train_fixed_assortments
                        num_hit = np.sum(hit == 1, axis=1)
                        print('Number of hits:', num_hit)
                        # count the ratio of num_zeros to 
                        num_ones = np.sum(test_fixed_assortments == 1, axis=1)
                        # print('Number of ones:', num_ones)
                        # calculate the ratio, if num_ones is 0, then that ratio be excluded
                        ratio = num_hit / num_ones # shape: (num_total_cells,)
                        # exclude the ratio where num_ones is 0
                        ratio = ratio[num_ones > 0]
                        # store the average value of the ratio for each test date
                        ratio = np.mean(ratio) 
                        print(f"Assortment hit ratio for {test_date}: {ratio:.4f}")
                        out_sample_assort_ratio[test_date] = ratio
                
                
                result = EstimationResult(f"{win_start}-{win_end}")
                result.time_window = f"{win_start}-{win_end}"
                result.p_hat = p_m
                result.omega = omega
                result.beta = beta
                result.lambda_hat = lambda_hat
                result.in_sample_wmapes = in_sample_wmapes
                result.out_sample_wmapes = out_sample_wmapes
                result.dist_per_grid = dist_per_grid
                result.in_relative_errors = in_sample_relative_errors 
                result.out_relative_errors = out_sample_relative_errors
                result.out_sample_assort_ratio = out_sample_assort_ratio
                result.in_o_ij_actual = in_o_ij_actual
                result.in_o_ij_hat = in_o_ij_actual
                result.in_N_hat = in_N_hat
                result.in_N_actual = in_N_actual
                result.out_o_ij_actual = out_o_ij_actual
                result.out_o_ij_hat = out_o_ij_hat
                result.dist = dist
                result.out_N_hat = out_N_hat
                result.out_N_actual = out_N_actual
                results.append(result)
        return results

    @staticmethod
    def visualize_results(results, save_dir):
        '''
        Visualize the results of temporal analysis across different time windows.
        
        Args:
            results: Dict with time window identifiers as keys and lists containing one EstimationResult object as values,
                    or a list of EstimationResult objects.
            save_dir: Directory path where the visualization plots will be saved.
        '''
        # Handle input types to ensure results is a list of EstimationResult objects
        if isinstance(results, dict):
            sorted_keys = sorted(results.keys())
            results = [results[key][0] if isinstance(results[key], list) else results[key] for key in sorted_keys]
        elif isinstance(results, list):
            results = sorted(results, key=lambda r: r.time_window)
        else:
            raise ValueError("results must be a list or dictionary of EstimationResult objects")

        # Extract number of cells and time windows
        num_total_cells = len(results[0].omega)
        windows = [r.time_window for r in results]

        # Plot alpha values
        alphas = [result.beta[0] for result in results]
        plt.figure(figsize=(10, 5))
        plt.plot(windows, alphas, marker='o')
        plt.title('Alpha Across Time Windows')
        plt.xlabel('Time Window')
        plt.ylabel('Alpha')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, 'alpha.png'))
        plt.close()

        # Plot v distribution
        v_dict = {r.time_window: r.beta[1:1+num_total_cells] for r in results}
        v_df = pd.DataFrame(v_dict)
        v_df.index.name = 'Location Index'
        plt.figure(figsize=(15, 8))
        for time_window in v_df.columns:
            plt.plot(v_df.index, v_df[time_window], marker='o', label=time_window)
        plt.title('v Distribution Across Location Indices')
        plt.xlabel('Location Index')
        plt.ylabel('v Value')
        plt.legend(title='Time Window', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'v_distribution.png'))
        plt.close()

        # Plot v0 distribution
        v0_dict = {r.time_window: r.beta[1+num_total_cells:] for r in results}
        v0_df = pd.DataFrame(v0_dict)
        v0_df.index.name = 'Location Index'
        plt.figure(figsize=(15, 8))
        for time_window in v0_df.columns:
            plt.plot(v_df.index, v0_df[time_window], marker='o', label=time_window)
        plt.title('v0 Distribution Across Location Indices')
        plt.xlabel('Location Index')
        plt.ylabel('v0 Value')
        plt.legend(title='Time Window', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'v0_distribution.png'))
        plt.close()

        # Plot v mean and std
        v_means = [np.mean(result.beta[1:1+num_total_cells]) for result in results]
        v_stds = [np.std(result.beta[1:1+num_total_cells]) for result in results]
        plt.figure(figsize=(10, 5))
        plt.errorbar(windows, v_means, yerr=v_stds, marker='o', capsize=5)
        plt.title('Mean and Std of v Across Time Windows')
        plt.xlabel('Time Window')
        plt.ylabel('v (mean ± std)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, 'v_summary.png'))
        plt.close()

        # Plot v0 mean and std
        v0_means = [np.mean(result.beta[1+num_total_cells:]) for result in results]
        v0_stds = [np.std(result.beta[1+num_total_cells:]) for result in results]
        plt.figure(figsize=(10, 5))
        plt.errorbar(windows, v0_means, yerr=v0_stds, marker='o', capsize=5)
        plt.title('Mean and Std of v0 Across Time Windows')
        plt.xlabel('Time Window')
        plt.ylabel('v0 (mean ± std)')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, 'v0_summary.png'))
        plt.close()

        # Plot lambda values
        plt.figure(figsize=(12, 6))
        lambdas = [r.lambda_hat for r in results]
        sns.barplot(x=windows, y=lambdas)
        plt.title('Lambda Values Across Time Windows')
        plt.ylabel('Lambda')
        plt.xticks(rotation=45)
        plt.savefig(os.path.join(save_dir, 'lambda_comparison.png'))
        plt.close()

        fig, ax = plt.subplots(figsize=(8, 6))
        lab_size = 15
        font_size = 15
        # Plot in-sample WMAPE
        # plt.figure(figsize=(12, 6))
        for train_date in set().union(*[r.in_sample_wmapes.keys() for r in results]):
            in_sample_wmapes = [r.in_sample_wmapes.get(train_date, np.nan) for r in results]
            sns.lineplot(x=windows, y=in_sample_wmapes, marker='o', label=f'In-sample WMAPE ({train_date})')
        plt.title('In-sample WMAPE Across Time Windows')
        ax.tick_params(labelsize=lab_size)
        # ax.set_xlabel("Ratio",fontsize=font_size)
        ax.set_ylabel("WMAPE (%)",fontsize=font_size)
        sns.set(font_scale=1)
        sns.set(style='white')
        sns.despine()
        # plt.ylabel('WMAPE (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'in_sample_wmape.png'))
        plt.close()

        # Plot out-of-sample WMAPE
        # plt.figure(figsize=(12, 6))
        fig, ax = plt.subplots(figsize=(8, 6))
        lab_size = 15
        font_size = 15
        for test_date in set().union(*[r.out_sample_wmapes.keys() for r in results]):
            out_sample_wmapes = [r.out_sample_wmapes.get(test_date, np.nan) for r in results]
            print('Out-sample WMAPE for', test_date, ':', out_sample_wmapes)
            sns.lineplot(x=windows, y=out_sample_wmapes, marker='o', label=f'Out-sample WMAPE ({test_date})')
        plt.title('Out-of-sample WMAPE Across Time Windows')
        ax.tick_params(labelsize=lab_size)
        # ax.set_xlabel("Ratio",fontsize=font_size)
        ax.set_ylabel("WMAPE (%)",fontsize=font_size)
        sns.set(font_scale=1)
        sns.set(style='white')
        sns.despine()
        # plt.ylabel('WMAPE (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'out_sample_wmape.png'))
        plt.close()

        # plot out-of-sample assortment ratio
        fig, ax = plt.subplots(figsize=(8, 6))
        lab_size = 15
        font_size = 15
        for test_date in set().union(*[r.out_sample_assort_ratio.keys() for r in results]):
            out_sample_assort_ratio = [r.out_sample_assort_ratio.get(test_date, np.nan) for r in results]
            sns.lineplot(x=windows, y=out_sample_assort_ratio, marker='o', label=f'Assortment hit ratio ({test_date})')
        plt.title('Out-of-sample Assortment Hit Ratio Across Time Windows')
        ax.tick_params(labelsize=lab_size)
        # ax.set_xlabel("Ratio",fontsize=font_size)
        ax.set_ylabel("Ratio (%)",fontsize=font_size)
        sns.set(font_scale=1)
        sns.set(style='white')
        sns.despine()
        # plt.ylabel('Ratio (%)')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'out_sample_assort_ratio.png'))
        plt.close()

        # Plot omega distribution
        omega_dict = {r.time_window: r.omega for r in results}
        omega_df = pd.DataFrame(omega_dict)
        omega_df.index.name = 'Location Index'
        plt.figure(figsize=(15, 8))
        for time_window in omega_df.columns:
            plt.plot(omega_df.index, omega_df[time_window], marker='o', label=time_window)
        plt.title('Omega Distribution Across Location Indices')
        plt.xlabel('Location Index')
        plt.ylabel('Omega Value')
        plt.legend(title='Time Window', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'omega_distribution.png'))
        plt.close()

        # Box plot for in-sample relative errors
        font_size = 15
        in_relative_errors_dict = {}
        in_mean = {}
        for r in results:
            errors = np.array(r.in_relative_errors)
            if len(errors) > 0:
                quantile_95 = np.quantile(errors, 0.95)
                filtered_errors = errors[errors <= quantile_95]
                in_relative_errors_dict[r.time_window] = filtered_errors
                in_mean[r.time_window] = np.mean(filtered_errors)
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
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'in_relative_errors.png'))
        plt.close()

        # Box plot for out-of-sample relative errors
        out_relative_errors_dict = {}
        out_mean = {}
        for r in results:
            errors = np.array(r.out_relative_errors)
            if len(errors) > 0:
                quantile_95 = np.quantile(errors, 0.95)
                filtered_errors = errors[errors <= quantile_95]
                out_relative_errors_dict[r.time_window] = filtered_errors
                out_mean[r.time_window] = np.mean(filtered_errors)
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
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'out_relative_errors.png'))
        plt.close()
if __name__ == "__main__":
    time_windows = [
        ('08:00', '10:00'),
        ('10:00', '12:00'),
        ('12:00', '14:00'),
        ('14:00', '16:00'),
        ('16:00', '18:00'),
        ('18:00', '20:00'),
        ('20:00', '22:00'),
        ('22:00', '23:59')
    ]
    training_dates = ['2022-10-17', '2022-10-18', '2022-10-19']
    testing_dates = ['2022-10-20', '2022-10-21']
    df = pd.read_csv('/home/go3/wch_code/jx/real_data/data/cleaned_data2.csv')
    analysis_results = TemporalAnalyzer.run_analysis(
        df=df,
        time_intervals=time_windows,
        dist_per_grid=3,
        timeperiod=30,
        training_dates=training_dates,
        testing_dates=testing_dates
    )
    save_dir = 'try'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    TemporalAnalyzer.visualize_results(analysis_results, save_dir)