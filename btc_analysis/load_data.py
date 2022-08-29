import requests
import json
import os
import time
from ast import literal_eval

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime as dt, timedelta

################################################################################
##                     LOADING DATA ABOUT ALL BLOCKS ON A DAY                  #
################################################################################

DIR_NAME = 'btc_tx_data'

## https://www.blockchain.com/api/blockchain_api
BLOCKCHAININFO_API_URL = 'https://blockchain.info'

# Retreive all blocks for a given date (ds)
def get_blocks(ds):
    ts = round((dt.strptime(ds,'%Y-%m-%d') + timedelta(days=1)).timestamp() * 1000)
    r = requests.get(f"{BLOCKCHAININFO_API_URL}/blocks/{ts}?format=json")
    if r.status_code == 404:
        raise Exception("Request failed")
    return r.json()

# Define data directory name
def get_data_dir(ds):
    return os.path.join(DIR_NAME, ds.replace('-', '_'))

# Define file name for batch of data
def get_fname(ds, n, batch=10):
    return os.path.join(get_data_dir(ds), f"tx_{n}_{n+batch-1}.json")

# Retrieve bitcoin data for ds and save locally ('batch' blocks at a time)
def download_transactions(ds, batch=10):
    # save data locally
    data_dir = get_data_dir(ds)
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    # get block data (ensure always sorted by block index)
    blocks = sorted(get_blocks(ds), key= lambda d: d['block_index'])

    # load transaction data
    txs = []
    n = 1
    print(f'====== Getting transactions for {len(blocks)} blocks on {ds} ======')
    while(n <= len(blocks)):
        if (os.path.exists(get_fname(ds, n, batch))):
            print(f"{n+batch-1} blocks already loaded")
            n+=batch
        else:
            h = blocks[n-1]['hash']
            r = requests.get(f"{BLOCKCHAININFO_API_URL}/block/{h}?format=json")

            if r.status_code == 404:
                raise Exception("Request failed")

            txs += r.json()['tx']

            if (n % batch == 0):
                # write to a file every 'batch' blocks
                with open(get_fname(ds, n-batch+1, batch), 'w') as f:
                    json.dump(txs, f)
                txs = []
                print(f"Completed {n} blocks")

            n+=1

# parse data from file
def extract_data(tx):
    return {'hash': tx['hash'],
            'time': tx['time'],
            'input_addr': sorted(list(filter(None, [i['prev_out'].get('addr') for i in tx['inputs']]))),
            'output_addr': sorted(list(filter(None, [o.get('addr') for o in tx['out']]))),
            # NOTE: value is in 10e-8 BTC
            'input_value_tot': sum([int(i['prev_out'].get('value')) for i in tx['inputs']]),
            'output_value_tot': sum([int(o.get('value')) for o in tx['out']]),
            'fee': tx['fee']}


# Get data from locally saved files for a given ds and return data frame
def get_transactions(ds):
    data_dir = get_data_dir(ds)

    if not os.path.isdir(data_dir):
        raise Exception("Data not downloaded yet.")

    txs = []

    for fn in os.listdir(data_dir):
        if('.json' in fn):
            with open(os.path.join(data_dir, fn), 'r') as f:
                txs += json.load(f)
    return pd.DataFrame([extract_data(tx) for tx in txs])

################################################################################
##                     LOADING DATA ABOUT SPECIFIC ADDRESSES                   #
################################################################################

## https://blockchair.com/api/docs#link_M2
BLOCKCHAIR_API_URL = 'https://api.blockchair.com/bitcoin/dashboards'

def fetch_addr_data(a):
    r = requests.get(f"{BLOCKCHAIR_API_URL}/address/{a}/?transaction_details=true&limit=100")

    if r.status_code == 429:
        raise Exception("Hit rate limit.")

    if r.status_code != 200:
        return None

    d = r.json()['data'][a]

    # get transaction receivd by the address
    times = [tx['time'] for tx in d['transactions'] if tx['balance_change'] > 0]

    return {'addr': a,
            'first_seen_receiving': d['address']['first_seen_receiving'],
            'total_received': d['address']['received'],
            'total_received_usd': d['address']['received_usd'],
            'lifetime_tx_count': d['address']['transaction_count'],
            'incoming_tx_times': times}

def load_all_addr_data(addrs, ds, label, batch=25, sleep_time=5):
    def get_addr_file(data_dir, n, batch):
        return os.path.join(data_dir, f'addr{int(np.floor(n/batch))}.csv')

    data_dir = os.path.join(get_data_dir(ds), f'addr_{label}')
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    n = 0
    n_err = 0
    addr_data = []

    print(f'===== Fetching data for {len(addrs)} addresses =====')
    while n < len(addrs):
        if os.path.exists(get_addr_file(data_dir, n, batch)):
            print(f"{n+batch} addresses already loaded")
            n+=batch

        else:
            data = fetch_addr_data(addrs[n])
            time.sleep(sleep_time) # delay 5 seconds to prevent rate limit (> 30 a minute)

            if data == None:
                n_err += 1
            else:
                addr_data += [data]

            if (n+1) % batch == 0:
                if 100 * n_err / batch >= 5:
                    print(f'Too many errors, failed on {n+1} addresses')
                else:
                    # save data frame to file
                    pd.DataFrame(addr_data).to_csv(get_addr_file(data_dir, n-1, batch))
                    print(f'Completed {n+1} addresses')
                n_err = 0
                addr_data = []
            n += 1

    print(f'Completed {n+1} addresses')

# get data from locally saved files on ds
def get_addresses(ds, label):

    def parse_time_list(s):
        l = literal_eval(s)
        return sorted([dt.strptime(t, '%Y-%m-%d %H:%M:%S') for t in l])

    data_dir = os.path.join(get_data_dir(ds), f'addr_{label}')

    if not os.path.isdir(data_dir):
        raise Exception("Data not downloaded yet.")

    addrs = []

    for fn in os.listdir(data_dir):
        if('.csv' in fn):
            with open(os.path.join(data_dir, fn), 'r') as f:
                d = pd.read_csv(f, index_col = 0)
                d.incoming_tx_times = d.incoming_tx_times.apply(parse_time_list)
                addrs += [d]

    return pd.concat(addrs).reset_index(drop=True)

def save_timing_df(df_test):
    df_test['arrival_times'] = df_test['arrival_times'].apply(lambda l: [dt.timestamp(t) for t in l])
    df_test.to_csv(os.path.join(DIR_NAME, 'df_timing.csv'))

def load_timing_df():
    df_test = pd.read_csv(os.path.join(DIR_NAME, 'df_timing.csv'), index_col=0)
    df_test['arrival_times'] = df_test['arrival_times'].apply(lambda l: [dt.fromtimestamp(t) for t in literal_eval(l)])
    return df_test

################################################################################
##                             ANALYZING BATCHING                             ##
################################################################################

# D is the length of a discrete time unit within which time we consider batching to have occurred
def get_batching_data(df, D=1, cvr=22e3):
    df['dt_time'] = [dt.fromtimestamp(t) for t in df['time']]
    df['n_outputs'] = [len(o) for o in df['output_addr']]
    df['n_inputs'] = [len(i) for i in df['input_addr']]
    df['output_addr_str'] = [','.join(addr) for addr in df['output_addr']]
    df['input_addr_str'] = [','.join(addr) for addr in df['input_addr']]

    # # rough btc to usd conversion
    cvr = 1e-8 * cvr
    df['input_value_tot_usd'] = round(df['input_value_tot'] * cvr, 2)
    df['output_vaue_tot_usd'] = round(df['output_value_tot'] * cvr, 2)

    # get start time, starting at a whole hour
    start_dt = df.dt_time.min().replace(second = 0, minute = 0)

    # assign create times to discrete epochs
    def get_epoch(ts, start, D):
        return int(np.floor((ts - start).total_seconds() / (60. * D)))

    df['epoch'] = [get_epoch(ts, start_dt, D) for ts in df['dt_time']]

    # filter out transactions with 0 input or output value or missing output or input addresses
    df =  df[(df.input_value_tot > 0)
                 & (df.output_value_tot > 0)
                 & (df.output_addr_str != '')
                 & (df.input_addr_str != '')]

    # get (output addr) x (epoch) where multiple input addr sent to the same output
    multiple = df.groupby(['output_addr_str', 'epoch'])['input_addr_str'].nunique()
    m = multiple[multiple > 1].index
    not_m = multiple[multiple == 1].index

    # get databases of batched and unbatched transactions
    df_batched = df.set_index(['output_addr_str', 'epoch']).loc[m].reset_index()
    df_unbatched = df.set_index(['output_addr_str', 'epoch']).loc[not_m].reset_index()

    df_batched['is_batched'] = True
    df_unbatched['is_batched'] = False

    return pd.concat([df_batched, df_unbatched]).reset_index(drop=True)
