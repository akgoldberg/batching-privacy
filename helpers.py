import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint,geom, uniform, bernoulli

import time
import os
import pickle
import json
import itertools

from multiprocess import Pool, cpu_count

################################################################################
############################# PARALLELIZE APPLIES ##############################
################################################################################
def applyParallel(df, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, df)
    return np.vstack(ret_list)

def applyParallelGroupBy(dfGrouped, func):
    with Pool(cpu_count()) as p:
        ret_list = p.map(func, [group for name, group in dfGrouped])
    return np.vstack(ret_list)

################################################################################
########################## NOISE ADDITION MECHANISMS ###########################
################################################################################

# accounts for epsilon / 2 needed in mechanism
def eps_to_K(eps):
    return 1./(1-np.exp(-eps/2.))

def sample_staircase(eps, g, gamma=None, size=1):
    # if gamma = None, use OPT
    if gamma is None:
        gamma = 1/(1+np.exp(eps/2.))
    b = np.exp(-1.*eps)
    G = geom.rvs(1-b, loc=-1, size=size)
    U = uniform.rvs(size=size)
    p0 = gamma / (gamma + ((1-gamma)*b))
    B = bernoulli.rvs(1.-p0, size=size)
    return (1-B)*((G + gamma*U)*g) + B*((G + gamma + ((1-gamma)*U))*g)

def apply_staircase(is_batched, g, eps, size=1):
    d = sample_staircase(eps/2., g, size=size)
    if is_batched:
        return g + d
    else:
        return d

def apply_uniform(is_batched, g, K, size=1):
    if is_batched:
        # sample from g to g*K
        loc = g
        scale = (K-1)*g
    else:
        # sample from 0 to g*K
        loc = 0
        scale = K*g
    return uniform.rvs(loc=loc, scale=scale, size=size)

# calculate expectation of zero-inflate uniform distribution
def expectation_unif(eps, g, batched, p = 1.):
    if batched:
        return g*(0.5 + (p / (2*(p - np.exp(-1*eps)))))
    else:
        return g*(p**2 / (2*(p-np.exp(-1*eps))))
