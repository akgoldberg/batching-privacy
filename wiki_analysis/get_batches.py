import pandas as pd
import numpy as np

import datetime as dt
import time
from functools import reduce
from multiprocess import Pool, cpu_count

def flatten(l):
    return reduce(lambda a,b: list(a) + list(b), l)

def xor(a,b):
    return a^b

def get_batches(g, cutoff=30):
    z = np.array(sorted(list(zip(g['revid'], g['dt_timestamp'])), key=lambda l:l[1]))
    ends = np.where(np.diff(z[:,1]) > dt.timedelta(minutes=cutoff))[0]
    s = 0
    batches = []
    for e in ends:
        batches += [z[s:e+1]]
        s = e+1

    if s < len(z):
        batches += [z[s:]]

    def assign_batch_id(b):
        # use first revid as batch_id
        return b[0][0]

    batch_ids = [assign_batch_id(b) for b in batches]

    batch_ids = flatten([np.repeat(batch_ids[i], len(b)) for i, b in enumerate(batches)])

    return np.vstack((z[:,0], batch_ids)).T

def applyParallel(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return np.vstack(ret_list)

def main():
    df = pd.read_csv('wiki_data_all.csv', index_col=0)
    df['dt_timestamp'] = df.timestamp.apply(lambda t: dt.datetime.strptime(t, '%Y-%m-%d %H:%M:%S'))
    g = df.groupby('user').pageid.nunique()
    df_g = df[df.user.isin(g[g>1].index)].groupby('user')[['revid', 'dt_timestamp']]
    # df_g = df[df.user.isin(df.user.unique()[:1000])].groupby('user')[['revid', 'dt_timestamp']]
    b = applyParallel(df_g, get_batches)
    with open('batches_15.npy', 'wb') as f:
        np.save(f, b)

if __name__ == "__main__":
    main()
