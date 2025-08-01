import numpy as np
from scipy.optimize import minimize
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
import os
import multiprocessing as mp
from collections import defaultdict

# Helper Functions
def compute_choice_probs(beta, d, S_mask, L, S):
    """
    Compute choice probabilities (p_{i,j} and p_{i,0}) using matrix operations.
    
    Parameters:
    - beta: Array containing v (utilities), v0 (no-purchase utilities), and alpha (distance sensitivity)
    - d: Distance matrix between grid cells
    - S_mask: Boolean mask indicating neighboring cells
    - L: Number of grid cells
    - S: List of neighbor indices for each cell
    
    Returns:
    - p_flat: Flattened array of choice probabilities
    """
    v = beta[:L]
    v0 = beta[L:2*L]
    alpha = beta[2*L]
    u_ij = v[np.newaxis, :] - alpha * d
    exp_u = np.exp(u_ij) * S_mask
    exp_v0 = np.exp(v0)
    denom = exp_v0[:, np.newaxis] + np.sum(exp_u, axis=1, keepdims=True)
    p_ij = exp_u / denom
    p_i0 = exp_v0[:, np.newaxis] / denom
    p_flat = []
    for i in range(L):
        p_flat.extend(p_ij[i, S[i]].tolist())
        p_flat.append(p_i0[i, 0])
    return np.array(p_flat)

def compute_wmape(true, est):
    """
    Compute Weighted Mean Absolute Percentage Error (WMAPE).
    
    Parameters:
    - true: True values
    - est: Estimated values
    
    Returns:
    - WMAPE value
    """
    abs_true = np.abs(true)
    if np.sum(abs_true) == 0:
        return np.nan
    return np.sum(np.abs(est - true)) / np.sum(abs_true)

def compute_rmse(true, est):
    """
    Compute Root Mean Squared Error (RMSE).
    
    Parameters:
    - true: True values
    - est: Estimated values
    
    Returns:
    - RMSE value
    """
    return np.sqrt(np.mean((est - true)**2))

def run_replication(task):
    """
    Run a single replication of the simulation for given grid size, time units, and replication index.
    
    Parameters:
    - task: Tuple (a, T, rep, E, d_bar, lambda_total)
        - a: Grid side length
        - T: Number of time units
        - rep: Replication index
        - E: Total area
        - d_bar: Maximum distance for neighbors
        - lambda_total: Total expected number of customers over the entire period
    
    Returns:
    - errors: Dictionary of WMAPE and RMSE for each parameter
    """
    a, T, rep, E, d_bar, lambda_total = task
    
    # Set seed for reproducibility
    np.random.seed(42 + rep)
    
    print(f"Running replication {rep + 1} for a={a}, T={T}")
    # Grid setup
    L = a * a  # Total number of cells
    grid_size = E / a  # Side length of each cell
    grid_centers = np.array([(i * grid_size + grid_size / 2, j * grid_size + grid_size / 2) 
                             for i in range(a) for j in range(a)])
    d = np.sqrt(np.sum((grid_centers[:, np.newaxis] - grid_centers[np.newaxis, :])**2, axis=2))
    S = [np.where(d[i] <= d_bar)[0] for i in range(L)]  # Neighbors within d_bar
    S_mask = np.zeros((L, L), dtype=bool)
    for i in range(L):
        S_mask[i, S[i]] = True
    
    # True parameters
    omega_true = dirichlet.rvs([1]*L)[0]  # Customer distribution
    v_true = np.array([np.sin(l / a) for l in range(1, L + 1)])  # Option utilities
    v0_true = np.random.normal(0, 0.1, L)  # No-purchase utilities
    alpha_true = 1  # Distance sensitivity
    beta_true = np.concatenate([v_true, v0_true, [alpha_true]])
    p_true = compute_choice_probs(beta_true, d, S_mask, L, S)
    
    # True lambda (arrival rate per time unit)
    lambda_true = lambda_total / T
    
    # Simulate data
    # n_i = np.random.poisson(lambda_total * omega_true, size=L)  # Total number of customers per cell over the entire period
    # D = []  # List of (origin cell, destination cell, time)
    # for i in range(L):
    #     u_i = v_true[S[i]] - alpha_true * d[i, S[i]]
    #     exp_u = np.exp(u_i)
    #     denom = np.exp(v0_true[i]) + np.sum(exp_u)
    #     probs = np.append(exp_u / denom, np.exp(v0_true[i]) / denom)
    #     choices = np.random.choice(np.append(S[i], -1), size=n_i[i], p=probs)
    #     for j in choices[choices != -1]:  # -1 indicates no purchase
    #         D.append((i + 1, j + 1, np.random.randint(T)))
    # N = len(D)  # Total number of purchases
    # N_total = np.random.poisson(lambda_total)  # Total number of customers over T time units
    # n_i = np.random.multinomial(N_total, omega_true)  # Distribute N_total customers across L cells
    D = []  # List of (origin cell, destination cell, time)
    for t in range(T):
        lambda_t = lambda_total/ T  # Arrival rate per time unit
        N_total_t = np.random.poisson(lambda_t)  # Total arrivals in this time unit
        n_t = np.random.multinomial(N_total_t, omega_true)
        for i in range(L):
            for _ in range(n_t[i]):
                # t = np.random.randint(T)  # Assign a random time unit from 0 to T-1
                u_i = v_true[S[i]] - alpha_true * d[i, S[i]]
                exp_u = np.exp(u_i)
                denom = np.exp(v0_true[i]) + np.sum(exp_u)
                probs = np.append(exp_u / denom, np.exp(v0_true[i]) / denom)
                choice = np.random.choice(np.append(S[i], -1), p=probs)
                if choice != -1:  # -1 indicates no purchase
                    D.append((i + 1, choice + 1, t))
    N = len(D)  # Total number of purchases
    
    # EM algorithm
    w = np.ones(L) / L  # Initial customer distribution
    beta = np.concatenate([np.random.uniform(0, 1, L), np.random.uniform(0, 1, L), [1]])  # Initial beta
    N_counts = np.zeros(L)
    for i_n, _, _ in D:
        N_counts[i_n - 1] += 1
    
    for _ in range(100):  # Max iterations
        v = beta[:L]
        v0 = beta[L:2*L]
        alpha = beta[2*L]
        u_ij = v[np.newaxis, :] - alpha * d
        exp_u = np.exp(u_ij) * S_mask
        exp_v0 = np.exp(v0)
        denom = exp_v0[:, np.newaxis] + np.sum(exp_u, axis=1, keepdims=True)
        p_ij = exp_u / denom
        p_i0 = exp_v0[:, np.newaxis] / denom
        p0_m = p_i0[:, 0]
        sum_wp0 = np.sum(w * p0_m)
        s = T * (1 - sum_wp0)
        if s <= 0:
            break
        c = N_counts + (N * T * w * p0_m) / s
        w_new = c / np.sum(c)
        
        def objective(beta_new):
            v_new = beta_new[:L]
            v0_new = beta_new[L:2*L]
            alpha_new = beta_new[2*L]
            u_ij_new = v_new[np.newaxis, :] - alpha_new * d
            exp_u_new = np.exp(u_ij_new) * S_mask
            exp_v0_new = np.exp(v0_new)
            denom_new = exp_v0_new[:, np.newaxis] + np.sum(exp_u_new, axis=1, keepdims=True)
            p_ij_new = exp_u_new / denom_new
            p_i0_new = exp_v0_new[:, np.newaxis] / denom_new
            log_p_sum = 0
            for i_n, j_n, _ in D:
                i, j = i_n - 1, j_n - 1
                log_p_sum += np.log(p_ij_new[i, j] + 1e-10)
            sum_no_purchase = np.sum(w * p0_m * np.log(p_i0_new[:, 0] + 1e-10))
            return -(log_p_sum + (N / s) * T * sum_no_purchase)
        
        res = minimize(objective, beta, method='L-BFGS-B')
        beta_new = res.x
        w_diff = np.linalg.norm(w_new - w)
        beta_diff = np.max(np.abs(beta_new - beta))
        if w_diff < 1e-4 and beta_diff < 1e-4:
            break
        w = w_new
        beta = beta_new
    else:
        print(f"EM did not converge for a={a}, T={T}, rep={rep}")
    
    # Estimates
    v_hat = beta[:L]
    v0_hat = beta[L:2*L]
    alpha_hat = beta[2*L]
    p_hat = compute_choice_probs(beta, d, S_mask, L, S)
    
    # Compute N_i_actual and N_i_hat
    N_i_actual = np.zeros(L)
    for i_n, _, _ in D:
        N_i_actual[i_n - 1] += 1
    p_i0_hat = np.array([p_hat[sum(len(S[k]) for k in range(i)) + len(S[i])] for i in range(L)])
    purchase_prob = 1 - p_i0_hat
    sum_over_l_b = np.sum(w * purchase_prob)
    lambda_hat = N / (T * sum_over_l_b) if sum_over_l_b > 0 else 0
    N_i_hat = lambda_hat * T * w * purchase_prob
    
    # Estimate lambda_total
    lambda_total_hat = lambda_hat * T
    
    # Compute errors
    errors = {}
    errors['omega'] = {'wmape': compute_wmape(omega_true, w), 'rmse': compute_rmse(omega_true, w)}
    errors['v'] = {'wmape': compute_wmape(v_true, v_hat), 'rmse': compute_rmse(v_true, v_hat)}
    errors['v0'] = {'wmape': compute_wmape(v0_true, v0_hat), 'rmse': compute_rmse(v0_true, v0_hat)}
    errors['choice_prob'] = {'wmape': compute_wmape(p_true, p_hat), 'rmse': compute_rmse(p_true, p_hat)}
    errors['alpha'] = {'wmape': compute_wmape(np.array([alpha_true]), np.array([alpha_hat])), 
                       'rmse': compute_rmse(np.array([alpha_true]), np.array([alpha_hat]))}
    errors['N_i'] = {'wmape': compute_wmape(N_i_actual, N_i_hat), 'rmse': compute_rmse(N_i_actual, N_i_hat)}
    errors['lambda_total'] = {'wmape': compute_wmape(np.array([lambda_total]), np.array([lambda_total_hat])), 
                              'rmse': compute_rmse(np.array([lambda_total]), np.array([lambda_total_hat]))}
    errors['lambda'] = {'wmape': compute_wmape(np.array([lambda_true]), np.array([lambda_hat])), 
                        'rmse': compute_rmse(np.array([lambda_true]), np.array([lambda_hat]))}
    print(f"FINISHED: Replication {rep + 1} complete for a={a}, T={T}")

    # Store errors into .npz file
    path = f'corrected_T/data/square_{a}x{a}_lambda_total_{lambda_total}/'
    os.makedirs(path, exist_ok=True)
    np.savez(f'{path}rep_{rep + 1}_errors.npz', **errors)

    # Return errors for this replication
    return errors

# Main Execution
if __name__ == '__main__':
    # Simulation parameters
    a_values = [
        20, 13, 10
        # , 8, 
        # 4
        # , 2
        ]  # Grid side lengths
    lambda_total = 4000  # Total expected number of customers over the entire period
    T_list = [1, 3, 5, 7, 9, 11]  # Number of time units
    num_replications = 2  # Number of replications per (a, T)
    E = 20  # Total area
    d_bar = 5  # Maximum neighbor distance
    
    # Create list of tasks
    tasks = [(a, T, rep, E, d_bar, lambda_total) 
             for a in a_values 
             for T in T_list 
             for rep in range(num_replications)]
    
    # Run simulations in parallel
    with mp.Pool(72) as pool:
        all_errors = pool.map(run_replication, tasks)
    
    # Aggregate results
    errors_by_a_T = defaultdict(lambda: defaultdict(list))
    task_idx = 0
    for a in a_values:
        for T in T_list:
            for rep in range(num_replications):
                errors = all_errors[task_idx]
                errors_by_a_T[a][T].append(errors)
                task_idx += 1
    
    # Average errors
    averaged_errors = {}
    for a in a_values:
        averaged_errors[a] = {}
        for T in T_list:
            averaged_errors[a][T] = {}
            for param in errors_by_a_T[a][T][0]:
                wmape_list = [errors[param]['wmape'] for errors in errors_by_a_T[a][T]]
                rmse_list = [errors[param]['rmse'] for errors in errors_by_a_T[a][T]]
                averaged_errors[a][T][param] = {
                    'wmape': np.mean(wmape_list),
                    'rmse': np.mean(rmse_list)
                }
    
    # Plot results
    parameters = ['omega', 'v', 'v0', 'choice_prob', 'alpha', 'N_i', 'lambda_total', 'lambda']
    param_labels = {
        'omega': 'Omega (Customer Distribution)',
        'v': 'v (Option Utilities)',
        'v0': 'v_0 (No-Purchase Utilities)',
        'choice_prob': 'Choice Probabilities',
        'alpha': 'Alpha (Distance Sensitivity)',
        'N_i': 'N_i (Orders per Grid)',
        'lambda_total': 'Lambda Total (Total Arrival Rate)',
        'lambda': 'Lambda (Arrival Rate per Time Unit)'
    }
    
    for a in a_values:
        T_values = T_list
        errors = averaged_errors[a]
        for param in parameters:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            lab_size = 15
            font_size = 15
            
            wmape_values = [errors[T][param]['wmape'] for T in T_values]
            rmse_values = [errors[T][param]['rmse'] for T in T_values]
            
            ax1.tick_params(labelsize=lab_size)
            ax1.plot(T_values, wmape_values, marker='o', color='b')
            ax1.set_title(f'WMAPE of {param_labels[param]}')
            ax1.set_xlabel('Number of Time Units (T)', fontsize=font_size)
            ax1.set_ylabel('Average WMAPE', fontsize=font_size)
            ax1.grid(True)
            
            ax2.plot(T_values, rmse_values, marker='o', color='r')
            ax2.set_title(f'RMSE of {param_labels[param]}')
            ax2.set_xlabel('Number of Time Units (T)', fontsize=font_size)
            ax2.set_ylabel('Average RMSE', fontsize=font_size)
            ax2.grid(True)
            
            plt.suptitle(f'Estimation Errors for {param_labels[param]} (Square Size {a}x{a}, lambda_total={lambda_total})')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            
            path = f'corrected_T/plots/square_{a}x{a}_lambda_total_{lambda_total}/'
            os.makedirs(path, exist_ok=True)
            plt.savefig(f'{path}square_{a}x{a}_lambda_total_{lambda_total}_{param}_errors.png')
            plt.close()
    
    print("All simulations complete.")