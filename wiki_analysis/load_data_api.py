import os
import sys
import gzip
import json
import threading

import datetime as dt
import time
import pandas as pd
import numpy as np

import pywikibot

SITE = pywikibot.Site('en', 'wikipedia')
START_DATE = dt.datetime(2022, 1, 1, 0, 0)
END_DATE = dt.datetime(2022, 2, 1, 0, 0)
SAVE_DIR = 'wiki_data_api'

COLS = ['user',
        'userid',
        'userhidden',
        'timestamp',
        'anon',
        'commenthidden',
        'revid',]


def get_all_pages():
    ## FROM: https://dumps.wikimedia.org/enwiki/20220601/
    with gzip.open("enwiki-20220601-all-titles-in-ns0.gz", "rb") as f:
        all_pages = [l.decode('utf-8').strip().replace('"', '') for l in f]

    print(f"Found {'{:,}'.format(len(all_pages)-1)} pages")

    return all_pages[1:]

def get_revs(page_name, start_date=START_DATE, site=SITE):
    try:
        page = pywikibot.Page(site, page_name)
        rgen = page.revisions(content=False, starttime=END_DATE, endtime=START_DATE)
    except pywikibot.exceptions.NoPageError:
        return None
    except pywikibot.exceptions.InvalidTitleError:
        return None
    except ValueError:
        return None

    o = list(rgen)
    if len(o) == 0:
        return None

    d = pd.DataFrame(o)[COLS]
    d['page'] = page_name
    d['pageid'] = page.pageid
    return d

def get_all_revs_threaded(page_list, n_threads=10, verbose=False):

    def get_revs_thread(page_list_split, results, i, verbose):
        t = time.time()
        revs=[]
        for n, p in enumerate(page_list_split):
            r = get_revs(p)
            if r is not None:
                revs.append(r)

            if n > 0 and n%1000 == 0 and verbose:
                print(f'Completed {n} iterations in {round(time.time() - t , 2)} seconds on thread {i}')
                t = time.time()

        if len(revs) > 0:
            results[i] = pd.concat(revs)

        if verbose:
            print(f'Finished thread {i}')

        return

    splits = np.array_split(np.array(page_list), n_threads)
    results = [None for i in range(len(splits))]
    threads = [None for i in range(len(splits))]

    for i, split in enumerate(splits):
        t = threading.Thread(target=get_revs_thread, args=(split, results, i, verbose,))
        threads[i] = t
        t.start()

    for thread in threads:
        thread.join()

    return pd.concat(results).reset_index(drop=True)

def load_data(pages, block=10000, save=100000, start=0):
    print(f"Loading data starting from {'{:,}'.format(start)}.")
    out = []
    N = len(pages)

    t = time.time()
    write_file = os.path.join(SAVE_DIR, f'data{start}_{start+save-1}.csv')
    for i in np.arange(start, N, block):
        if i%save == 0 and i > start:
            pd.concat(out).reset_index(drop=True).to_csv(write_file)
            out = []
            print(f'Saved downloaded data to {write_file}')
            write_file = os.path.join(SAVE_DIR, f'data{i}_{i+save-1}.csv')
        l = pages[i:min(i+block, N)]
        o = get_all_revs_threaded(l)
        out += [o]
        print(f'Loaded pages {i} to {min(i+block-1, N)} in {round(time.time() - t, 2)} seconds')


    pd.concat(out).reset_index(drop=True).to_csv(write_file)

def combine_data(out='wiki_data_all.csv'):
    ds = []
    for fn in os.listdir(SAVE_DIR):
        if('.csv' in fn):
            with open(os.path.join(SAVE_DIR, fn), 'r') as f:
                d = pd.read_csv(f, index_col = 0)
                ds += [d]

    df = pd.concat(ds)
    df.to_csv('wiki_data_all.csv')

def main():
    args = sys.argv[1:]
    if args[0] == '--start':
        start = int(args[1])
        assert(start >= 0)
    else:
        start=0

    all_pages = get_all_pages()
    load_data(all_pages, start=start)

    combine_data()

if __name__ == "__main__":
    main()
