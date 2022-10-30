import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import time
import os
import pickle
import json
import itertools

from multiprocess import Pool, cpu_count

import sys
sys.path.append('..')
from helpers import *

import load_data

# set default g-value of 10 minutes when not enough data to evaluate
DIR_NAME = 'btc_tx_data'
DEFAULT_G = 10 * 60

P_vals = [25, 50, 75, 95]
eps_vals = [0.1, 0.5, 1., 2.]
Ts = np.hstack((np.arange(1, 10, 2), np.arange(10, 60, 10), np.arange(60, 3600, 300), np.arange(3600, 200000, 50000)))
Ws = np.hstack((np.arange(0.01, 0.1, 0.02), np.arange(0.1, 0.5, 0.1), np.arange(0.5, 6.5, 0.5), np.arange(6.5, 20, 3)))

################################################################################
################################# EXPERIMENTS ##################################
################################################################################

# Load data from csv and prepare for analysis
def load_and_clean_data(batch_cutoff=60):
    df_test = load_data.load_timing_df()
    df_test['n_txs_past'] = df_test.arrival_times.apply(len)
    df_test['has_tx_history'] = df_test['n_txs_past'] >= 2

    # Filter out likely exchanges
    past_tx_distr = df_test[df_test.n_txs_past > 0].groupby('output_addr_str').output_value_totusd_historical.max()
    m_historical = np.percentile(past_tx_distr[past_tx_distr > 0], 99)
    past_tx_distr_n = df_test.groupby('output_addr_str').n_txs_past.max()
    m_tx_historical = np.percentile(past_tx_distr_n[past_tx_distr_n > 0], 99)

    # Get batches
    df_pairs = get_pairs(df_test)
    # Batches are to the same output addr within one minute of one another
    df_batches = df_pairs[df_pairs.time_diff <= batch_cutoff]
    batched_hashes = set(df_batches.hash_1).union(set(df_batches.hash_2))
    df_test['is_batched'] = False
    df_test.loc[df_test.hash.isin(batched_hashes), 'is_batched'] = True

    return df_test, df_pairs

# get all pairs of tx sent to the same output address and the times
def get_pairs(df, time_col='time', g_cols=[]):
    cols = ['output_addr_str', time_col, 'hash'] + g_cols
    df_tmp = df[cols].set_index('output_addr_str')
    df_pairs = df_tmp.join(df_tmp, how='inner', lsuffix = '_1', rsuffix='_2').reset_index()
    df_pairs = df_pairs[df_pairs.hash_1 < df_pairs.hash_2]
    df_pairs['time_diff'] = np.abs(df_pairs[f'{time_col}_1'] - df_pairs[f'{time_col}_2'])
    return df_pairs.sort_values(by=['hash_1', 'hash_2'])

# compute g based on 7d tx arrival history to the same output address
def compute_g_percentile(times, Ps, batch_cutoff=60):
    diffs = np.diff(times)
    n_batched = sum(diffs <= batch_cutoff)
    if n_batched == len(diffs):
        return [DEFAULT_G for p in Ps]
    else:
        # take percentile removing batches
        return [np.percentile(diffs[diffs >= batch_cutoff], p) for p in Ps]

# sample uniform noise or staircase noise for specific value of g and K/eps
def sample_noise(df, g_col, apply_distr, privacy_param, iters):
    # sample uniforms for everything using default g
    defaults_index = df[~df.is_batched & (df[g_col] == DEFAULT_G)].index
    defaults_values = apply_distr(False, DEFAULT_G, privacy_param, len(defaults_index)*iters)
    defaults = pd.DataFrame(data=defaults_values.reshape(-1,iters), index=defaults_index)

    # sample uniforms for different g's
    df_non_default = df[~df.index.isin(defaults_index)]
    distr = lambda r: apply_distr(r['is_batched'], r[g_col], privacy_param, iters)
    non_defaults_values = df_non_default.apply(distr, axis=1).values
    non_defaults_values = np.concatenate(non_defaults_values).reshape(-1, iters)
    non_defaults = pd.DataFrame(data=non_defaults_values, index=df_non_default.index)

    return pd.concat([non_defaults, defaults])

# draw random delay from staircase and uniform distributions (iters times)
def generate_delays(df, eps, P_vals, iters=5, batched_only=False):
    if batched_only:
        df = df[df.is_batched]

    out = {}
    for p in P_vals:
        unif_runs = sample_noise(df, f'g_{p}', apply_uniform, eps_to_K(eps), iters)
        stair_runs = sample_noise(df, f'g_{p}', apply_staircase, eps, iters)
        out[f'{p}'] = {'uniform': unif_runs, 'staircase': stair_runs}
    return out


# simulate informed and basic thersholding attack with no privacy noise added
def simulate_attack_no_privacy(df_pairs, P_val):
    batch = df_pairs['time_diff'] <= 60
    out_basic = []
    out_informed = []
    g = df_pairs[f'g_{P_val}']

    for T in Ts:
        batch_pred = df_pairs['time_diff'] <= T
        n_TP = sum(batch_pred & batch)
        n_FP = sum(batch_pred & ~batch)
        n_FN = sum(~batch_pred & batch)
        out_basic += [(n_TP, n_FP, n_FN)]

    for W in Ws:
        batch_pred = df_pairs['time_diff'] <= g*W

        n_TP = sum(batch_pred & batch)
        n_FP = sum(batch_pred & ~batch)
        n_FN = sum(~batch_pred & batch)
        out_informed += [(n_TP, n_FP, n_FN)]

    return out_basic, out_informed

# simulate informed and basic thresholding attack for one setting of g and various eps
def simulate_attack(df, df_pairs, noise, P_val, eps_vals, iters=5):
    df_noise_unif = pd.concat([noise[f'{eps}'][f'{P_val}']['uniform'] for eps in eps_vals], axis=1)
    cols = np.arange(0, iters*len(eps_vals))
    df_noise_unif.columns = cols
    df_noise_unif = df_noise_unif.join(df[['hash', 'time', 'is_batched', f'g_{P_val}']])
    # turn into release times
    for c in cols:
        df_noise_unif[c] = df_noise_unif[c] + df_noise_unif['time']

    # get release times per pair
    df_pairs_noise_1 = pd.merge(df_noise_unif, df_pairs, left_on='hash', right_on='hash_1')
    df_pairs_noise_2 = pd.merge(df_noise_unif, df_pairs, left_on='hash', right_on='hash_2')
    df_pairs_noise_1 = df_pairs_noise_1.sort_values(by=['hash_1', 'hash_2'] , ignore_index=True)
    df_pairs_noise_2 = df_pairs_noise_2.sort_values(by=['hash_1', 'hash_2'], ignore_index=True)

    batch = df_pairs_noise_1['time_diff'] <= 60
    out_basic = []
    out_informed = []
    g = df_pairs_noise_1[f'g_{P_val}']

    out_basic_nopriv, out_informed_nopriv = simulate_attack_no_privacy(df_pairs_noise_1, P_val)

    out_basic += out_basic_nopriv
    out_informed += out_informed_nopriv

    for c in cols:
        time_diff_priv = np.abs(df_pairs_noise_1[c] - df_pairs_noise_2[c])

        for T in Ts:
            batch_pred = time_diff_priv <= T
            n_TP = sum(batch_pred & batch)
            n_FP = sum(batch_pred & ~batch)
            n_FN = sum(~batch_pred & batch)
            out_basic += [(n_TP, n_FP, n_FN)]

        for W in Ws:
            batch_pred = time_diff_priv <= g*W

            n_TP = sum(batch_pred & batch)
            n_FP = sum(batch_pred & ~batch)
            n_FN = sum(~batch_pred & batch)
            out_informed += [(n_TP, n_FP, n_FN)]

    return out_basic, out_informed

# Do all loading of data and generation of g values
def load_data_with_g(P_vals, eps_vals):
    df_base_file_path = os.path.join(DIR_NAME, 'df_base.csv')
    if os.path.exists(df_base_file_path):
        t=time.time()
        print('Loading data from saved')
        df_base = pd.read_csv(df_base_file_path, index_col=0)
        df_pairs = get_pairs(df_base)

        print(f'Completed loading full dataset from saved in {time.time() - t} seconds')
        print(f'Found {df_base.shape[0]} total transactions')
        print(f'Found {df_base.is_batched.sum()} batched transactions')
    else:
        t = time.time()
        df_base, df_pairs = load_and_clean_data()
        print(f'Completed loading data in {time.time() - t} seconds')
        print(f'Found {df_base.shape[0]} total transactions')
        print(f'Found {df_base.is_batched.sum()} batched transactions')

        # Compute g for different percentiles
        t = time.time()
        # g_Ps = applyParallel(df_base.arrival_times, lambda t: compute_g_percentile(t, P_vals))
        g_Ps = df_base.arrival_times.apply(lambda t: compute_g_percentile(t, P_vals))
        df_base[[f'g_{p}' for p in P_vals]] = pd.DataFrame(g_Ps.tolist())
        df_base.to_csv(df_base_file_path)

        print(f'Completed computing g values in {time.time() - t} seconds')

    # Generate random delays for combinations of epsilon and g if not already generated
    t = time.time()
    delay_file_path = os.path.join(DIR_NAME, 'saved_delays.pkl')
    if os.path.exists(delay_file_path):
        with open(delay_file_path, 'rb') as f:
            res = pickle.load(f)
    else:
        res = {}
        for eps in eps_vals:
            t_last = time.time()
            res[f'{eps}'] = generate_delays(df_base, eps, P_vals)
            print(f'Generated random delays for epsilon={eps} in {time.time() - t_last} seconds')

        with open(delay_file_path, 'wb') as f:
            pickle.dump(res, f)
    print(f'Generated random delays in {time.time() - t} seconds')

    return df_base, df_pairs, res

def get_attack_data(out, n_tot, iters, kind):
    if kind == 'basic':
        n_params = len(Ts)
    if kind == 'informed':
        n_params = len(Ws)

    out_no_priv = out[:n_params]
    out = out[n_params:]
    n_TP_0 = np.array([l[0] for l in out_no_priv])
    n_FP_0 = np.array([l[0] for l in out_no_priv])
    n_FN_0 =  np.array([l[2] for l in out_no_priv])
    n_TN_0 = n_tot - (n_TP_0 + n_FP_0 + n_FN_0)

    n_TP = np.array([l[0] for l in out]).reshape(-1, iters, n_params).sum(axis=1)
    n_FP = np.array([l[1] for l in out]).reshape(-1, iters, n_params).sum(axis=1)
    n_FN =  np.array([l[2] for l in out]).reshape(-1, iters, n_params).sum(axis=1)
    n_TN = iters*n_tot - (n_TP + n_FP + n_FN)

    return [np.vstack([n_TP_0, n_TP]), np.vstack([n_FP_0, n_FP]), np.vstack([n_TN_0, n_TN]), np.vstack([n_FN_0, n_FN])]


def load_experiment_results(P_vals_to_load, iters=5):
    df_base, df_pairs, res = load_data_with_g(P_vals, eps_vals)

    df_noise_unif = pd.concat([res[f'{eps}'][f'{P}']['uniform'] for eps,P in itertools.product(eps_vals, P_vals)], axis=1)
    cols = np.arange(0, iters*len(eps_vals)*len(P_vals))
    df_noise_unif.columns = cols
    df_noise_unif = df_noise_unif.join(df_base[['hash', 'time', 'is_batched'] + [f'g_{P}' for P in P_vals]])

    df_noise_staircase = pd.concat([res[f'{eps}'][f'{P}']['staircase'] for eps,P in itertools.product(eps_vals, P_vals)], axis=1)
    df_noise_staircase.columns = cols
    df_noise_staircase = df_noise_staircase.join(df_base[['hash', 'time', 'is_batched'] + [f'g_{P}' for P in P_vals]])

    a = df_noise_unif[df_noise_unif.is_batched][cols].to_numpy()
    unif_noise = np.hstack(a.reshape(-1, len(P_vals) * len(eps_vals), iters).T)

    a = df_noise_staircase[df_noise_staircase.is_batched][cols].to_numpy()
    staircase_noise = np.hstack(a.reshape(-1, len(P_vals) * len(eps_vals), iters).T)

    results_basic = {}
    results_informed = {}
    for P_val in P_vals_to_load:
        attack_file_path = os.path.join(DIR_NAME, f'saved_attack_res_{P_val}.pkl')
        with open(attack_file_path, 'rb') as f:
            out_basic, out_informed = pickle.load(f)
        # return four lists (TP, FP, FN, TN) with list of results for attack for no privacy and each epsilon value
        results_basic[P_val] = get_attack_data(out_basic, df_pairs.shape[0], iters, kind='basic')
        results_informed[P_val] = get_attack_data(out_informed, df_pairs.shape[0], iters, kind='informed')

    # return delay distributions and attack results
    return unif_noise, staircase_noise, results_basic, results_informed

def main():
    args = sys.argv[1:]
    if args[0] is None:
        print("Missing value of percentile to test experiments on.")
        return
    else:
        P_to_test = int(args[0])
        print(f'Running experiments using {P_to_test}th percentile')

    # Load data
    df_base, df_pairs, res = load_data_with_g(P_vals, eps_vals)

    # Simulate attacks
    t = time.time()
    out_basic, out_informed = simulate_attack(df_base, df_pairs, res, P_to_test, eps_vals)
    out_basic
    attack_file_path = os.path.join(DIR_NAME, f'saved_attack_res_{P_to_test}.pkl')
    with open(attack_file_path, 'wb') as f:
        pickle.dump([out_basic, out_informed], f)
    print(f'Ran informed attack once in {time.time() - t} seconds')

if __name__ == "__main__":
    main()

# ################################################################################
# #################################### OLD #######################################
# ################################################################################
#
# def plot_delays(stair_runs, epoch_runs, cutoff=95, ax=None):
#     if ax == None:
#         fig, ax = plt.subplots(figsize=(10,10))
#
#     s = np.hstack(stair_runs)
#     e = np.hstack(epoch_runs)
#
#     x_max = 300
#
#     ax.hist([e, s], bins=np.arange(0, x_max, 10), alpha=0.9, label=['random epoch', 'staircase'], density=True)
#
#     ax.legend(fontsize=16)
#
#     ax.set_xlabel('Number of Minutes Delayed')
#     ax.set_ylabel('Density')
#
# def get_percentiles(stair_runs, epoch_runs, eps_value):
#     p25 = np.percentile(stair_runs, 25), np.percentile(epoch_runs, 25)
#     p50 = np.percentile(stair_runs, 50), np.percentile(epoch_runs, 50)
#     p75 = np.percentile(stair_runs, 75), np.percentile(epoch_runs, 75)
#     p95 = np.percentile(stair_runs, 95), np.percentile(epoch_runs, 95)
#     avg = np.mean(stair_runs), np.mean(epoch_runs)
#
#     d = pd.DataFrame([p25, p50, p75, p95, avg]).T
#     d.index = ['Staircase', 'Random Epoch']
#     d.columns= ['p25', 'p50', 'p75', 'p95', 'Mean']
#     eps = r'$\epsilon$'
#     d[eps] = eps_value
#     d.set_index(eps, append=True, inplace=True)
#     return d.round(decimals=1)
