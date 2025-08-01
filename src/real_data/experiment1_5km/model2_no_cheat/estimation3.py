import numpy as np
from scipy.optimize import minimize
from math import radians, cos, sin, asin, sqrt
from preprocessing3 import Preprocessor
import pdb

class EstimationResult:
    def __init__(self, time_window):
        self.time_window = time_window
        self.p_hat = None
        self.omega = None
        self.beta = None
        self.lambda_hat = None
        self.in_sample_wmapes = None
        self.out_sample_wmapes = None
        self.dist_per_grid = None
        self.in_relative_errors = None
        self.out_relative_errors = None
        self.out_sample_assort_ratio = None
        self.in_o_ij_actual = None
        self.in_o_ij_hat = None
        self.dist = None
        self.in_N_hat = None
        self.in_N_actual = None
        self.out_o_ij_actual = None
        self.out_o_ij_hat = None
        self.out_N_hat = None
        self.out_N_actual = None

class Estimator:
    @staticmethod
    def haversine(lng1, lat1, lng2, lat2):
        lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
        dlng = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlng/2)**2
        c = 2 * asin(sqrt(a))
        return c * 6371

    @staticmethod
    def compute_p(beta, dist, num_total_cells, fixed_assortments):
        alpha = beta[0]
        v = beta[1:1 + num_total_cells]
        v0 = beta[1 + num_total_cells:]

        exp_utilities = np.exp(v - alpha * dist)
        exp_no_purchase = np.exp(v0)

        exp_utilities = exp_utilities * fixed_assortments
        
        denominator = exp_no_purchase + np.sum(exp_utilities, axis=1)
        p_no_purchase = exp_no_purchase / denominator
        p_purchase = exp_utilities / denominator[:, None]
        
        p = np.hstack((p_purchase, p_no_purchase[:, None]))
        return p

    @staticmethod
    def objective_beta(beta, cust_idx, sell_idx, omega, p_m, factor, dist, num_total_cells, fixed_assortments, penalty=0):
        try:
            p = Estimator.compute_p(beta, dist, num_total_cells, fixed_assortments)
            log_p_purchase = np.log(p[cust_idx, sell_idx] + 1e-15)
            log_p_no_purchase = np.log(p[:, -1] + 1e-15)
            p_no_purchase = p[:, -1] + 1e-15
            reg = penalty * np.sum(beta**2)
            return -(np.sum(log_p_purchase) + factor * np.sum(p_no_purchase * omega * log_p_no_purchase) - reg)
        except:
            return np.inf

    @staticmethod
    def pre_process(df, num_cells_lon, dist_per_grid, num_total_cells):
        dist = np.zeros((num_total_cells, num_total_cells))
        for i in range(num_total_cells):
            index_i = i
            lat_index_i, lng_index_i = Preprocessor.get_latlng_from_id(index_i, num_cells_lon)
            for j in range(num_total_cells):
                index_j = j
                lat_index_j, lng_index_j = Preprocessor.get_latlng_from_id(index_j, num_cells_lon)
                dist[i, j] = np.sqrt((np.abs(lat_index_i - lat_index_j) * dist_per_grid) ** 2 + 
                                     (np.abs(lng_index_i - lng_index_j) * dist_per_grid) ** 2)
        return df, dist

    @staticmethod
    def EM(dist_per_grid, df, dist, num_total_cells, fixed_assortments, max_iter=200, tol=1e-5, penalty=0):
        N = len(df)
        omega = np.ones(num_total_cells) / num_total_cells
        beta = np.zeros(1 + num_total_cells + num_total_cells)
        beta[0] = 10
        beta[1:1 + num_total_cells] = 0
        beta[1 + num_total_cells:] = 0
        
        for iteration in range(max_iter):
            p_m = Estimator.compute_p(beta, dist, num_total_cells, fixed_assortments)
            sum_p_no_purchase = (omega * p_m[:, -1]).sum()
            if (1 - sum_p_no_purchase) < 1e-6:
                break
            factor = N / (1 - sum_p_no_purchase)
            c = np.array([(df['cust_index'] == i ).sum() + factor * omega[i] * p_m[i, -1] 
                          for i in range(num_total_cells)])
            omega_new = c / (c.sum())
            res = minimize(
                Estimator.objective_beta, beta,
                args=(df['cust_index'].values, df['sell_index'].values, omega, p_m, factor, dist, 
                      num_total_cells, fixed_assortments, penalty),
                method='L-BFGS-B',
                bounds=[(0, None)] + [(None, None)] * num_total_cells + [(None, None)] * num_total_cells
            )
            beta_new = res.x
            rate_change_beta = np.max(np.abs(beta_new - beta) / (1000 * (beta + 1e-10)))
            print(f'dist_per_grid: {dist_per_grid}| iteration: {iteration} | diff_omega: {np.max(np.abs(omega_new - omega))} | diff_beta: {rate_change_beta}')
            if (np.max(np.abs(omega_new - omega)) < tol) and (rate_change_beta < tol):
                print(f"Converged at iteration {iteration}")
                break
            omega, beta = omega_new, beta_new
            print(f"Iter {iteration}: Loss {res.fun:.2f}, Alpha {beta[0]:.3f}")
        return omega, beta
    
    @staticmethod
    def estimate_parameters(num_total_cells, num_cells_lon, window_df, dist_per_grid=0.5, timeperiod=5, max_iter=200, tol=1e-4, fixed_assortments = None, penalty=0):
        # window_df, num_total_cells, num_cells_lon, _, _, _, _ = Preprocessor.cut_df(window_df, dist_per_grid, timeperiod)
        df_processed, dist = Estimator.pre_process(window_df, num_cells_lon, dist_per_grid, num_total_cells)
        # fixed_assortments = Preprocessor.compute_fixed_assortments(df_processed, dist, num_total_cells)
        omega, beta = Estimator.EM(dist_per_grid, df_processed, dist, num_total_cells, fixed_assortments, max_iter=max_iter, 
                                   tol=tol, penalty=penalty)
        p_m = Estimator.compute_p(beta, dist, num_total_cells, fixed_assortments)
        T_train = df_processed['time_index'].nunique()
        sum_over_l_b = (omega[:, None] * p_m[:, :-1]).sum()
        d_eval = df_processed['date'].nunique()
        lambda_hat = len(df_processed) / (d_eval * T_train * sum_over_l_b) if T_train * sum_over_l_b != 0 else 0
        return omega, beta, lambda_hat, dist, p_m

    @staticmethod
    def calculate_wmape(extra_assortment, p_m, omega, beta, lambda_hat, dist, eval_df, dist_per_grid, timeperiod):
        eval_df, num_total_cells, _, _, _, _, _ = Preprocessor.cut_df(eval_df, dist_per_grid, timeperiod)
        T_eval = eval_df['time_index'].nunique()
        
        # if extra_assortment == True:
        # special note:
        p_m = Estimator.compute_p(beta, dist, num_total_cells, extra_assortment)

        # Calculate actual o_ij
        o_ij_actual = np.zeros((num_total_cells, num_total_cells))
        for _, row in eval_df.iterrows(): # 
            i = row['cust_index']  # Adjust to 0-based index
            j = row['sell_index']  # Adjust to 0-based index
            # calculate choose probability for each i, j
            o_ij_actual[i, j] += 1
        # pdb.set_trace()
        # o_ij_actual = o_ij_actual / o_ij_actual.sum(axis=1, keepdims=True) if o_ij_actual.sum(axis=1, keepdims=True) > 0 else 0

        # Calculate predicted o_ij
        o_ij_hat = np.zeros((num_total_cells, num_total_cells))
        for i in range(num_total_cells):
            for j in range(num_total_cells):
                # calculate choose probability for each i, j
                o_ij_hat[i, j] = lambda_hat * T_eval * omega[i] * p_m[i, j]
            # normalize each row to sum to 1
            o_ij_actual[i, :] = o_ij_actual[i, :] / o_ij_actual[i, :].sum() if o_ij_actual[i, :].sum() > 0 else 0
            o_ij_hat[i, :] = o_ij_hat[i, :] / o_ij_hat[i, :].sum() if o_ij_hat[i, :].sum() > 0 else 0

        # pdb.set_trace()
        # o_ij_hat = o_ij_hat / o_ij_hat.sum(axis=1, keepdims=True) if o_ij_hat.sum(axis=1, keepdims=True) > 0 else 0

        # Calculate relative error for each i, j where o_ij > 0
        # relative_errors = []
        # for i in range(num_total_cells):
        #     for j in range(num_total_cells):
        #         # if o_ij_actual[i, j] > 0:
        #         error = abs(o_ij_hat[i, j] - o_ij_actual[i, j]) / (o_ij_actual[i, j] + 1e-6) * 100
        #         relative_errors.append(error)
        
        # flatten the arrays to 1D
        o_ij_actual = o_ij_actual.flatten()
        o_ij_hat = o_ij_hat.flatten()

        # Calculate relative errors, only division by o_ij_actual where o_ij_actual > 0
        # relative_errors = np.where(o_ij_actual > 0, (np.abs(o_ij_actual - o_ij_hat) / o_ij_actual) * 100, 0)
        # turn off the error report for division by zero
        # relative_errors = np.where(o_ij_actual > 0, (np.abs(o_ij_actual - o_ij_hat) / o_ij_actual) * 100, 0)
        
        relative_errors = np.abs(o_ij_actual - o_ij_hat)

        # remove inf values from relative_errors away
        relative_errors = relative_errors[~np.isinf(relative_errors) & ~np.isnan(relative_errors)]  # remove inf values
        # print(f'Relative errors: {relative_errors}')
        # relative_errors = (np.abs(o_ij_actual - o_ij_hat) / o_ij_actual) * 100  if o_ij_actual > 0 else 0


        # Calculate WMAPE as before
        N_hat = np.array([lambda_hat * T_eval * (omega * p_m[:, c]).sum() for c in range(num_total_cells)])
        N_actual = np.zeros(num_total_cells)
        for i in range(num_total_cells):
            N_actual[i] = len(eval_df[(eval_df['sell_index'] == i)])
        wmape = (np.abs(N_actual - N_hat).sum() / N_actual.sum()) * 100 if N_actual.sum() > 0 else 0 # shape: 
        # print(f'N_hat: {N_hat} | N_actual: {N_actual}\nT_eval: {T_eval} | wmape: {wmape}')
        
        return wmape, relative_errors, o_ij_actual, o_ij_hat, dist, N_hat, N_actual