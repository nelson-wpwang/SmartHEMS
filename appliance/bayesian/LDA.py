from __future__ import division

from matplotlib import pyplot as plt
import numpy as np
import pymc3 as pm
import scipy as sp
import seaborn as sns
import theano
from theano import tensor as tt
import pandas as pd
import os
import pickle
import time

SEED = 123
W = 7
K = 10
with open('./data/dataset.pkl', 'rb') as f:
    data = np.array(pickle.load(f))
data = data.transpose([1, 0, 2])
Q = np.squeeze(data[:31, 0, -1])
data = data[:31, :, :-1]
L = 6
T = 24


# Hyper_parameters
alpha = .5
beta1 = 2
beta2 = 5
omega = np.array([2., 2., 2., 2., 2., 2., 2.])

data_tensor = theano.shared(data)
Q_tensor = theano.shared(Q)

def stick_breaking(beta):
    portion_remaining = tt.concatenate([[1], tt.extra_ops.cumprod(1 - beta)[:-1]])

    return beta * portion_remaining

def get_Pi_star(d, Pi, M, Theta):
    ups = []
    for k in range(K):
        likelihood = data_tensor[d, :, :] ** M[k, : , :] * (1 - data_tensor[d, :, :]) ** (1 - M[k, : , :])
        # ha = data_tensor[d, :, :] ** M[k, : , :] * (1 - data_tensor[d, :, :]) ** (1 - M[k, : , :]) * Pi[k]
        up = Pi[k] * likelihood.prod() * Theta[k, Q_tensor[d]]
        up = up.dimshuffle('x')
        # up = Pi[k] * theano.power(data_tensor[d, :, :], M[k, : , :]) * theano.power(1 - data_tensor[d, :, :], 1 - M[k, : , :]) * Theta[k, Q_tensor[d]]
        ups.append(up)
    ups = tt.concatenate(ups, axis=0)
    return ups / tt.sum(ups)


start = time.time()
with pm.Model() as model:
    '''
    From lambda to theta_k
    '''
    Theta = pm.Dirichlet('Theta', a=omega, shape=(K, W))
    # print(Theta[9, 6])

    '''
    From alpha to Pi
    '''
    # Latent parameters
    V = pm.Beta('V', 1., alpha, shape=K)
    Pi = pm.Deterministic('Pi', stick_breaking(V))

    '''
    From beta to mu
    '''
    M = pm.Beta('M', beta1, beta2, shape=(K, L, T))

    '''
    From z, theta, and mu to obs
    '''
    for d in range(data.shape[0]):
        print('Builidng model for day [%d]'%d, end='\r')
        Pi_star = pm.Deterministic(f'Pi_star_{d}', get_Pi_star(d, Pi, M, Theta))
        z = pm.Categorical(f"Z_{d}", p=Pi_star)
        theta = Theta[z]
        obs2 = pm.Categorical(f'Q_{d}', p=theta, observed=Q[d])
        for l in range(data.shape[1]):
            for t in range(data.shape[2]):
                obs1 = pm.Bernoulli(f'obs_{d,l,t}', p=M[z, l, t], observed=data[d, l, t])

print('\n')
print('Finish building my model in %.2f seconds'%(time.time() - start))

with model:
    trace = pm.sample(10, random_seed=SEED, init='advi', progressbar=True)

def print_mu(k, l):
    return trace['M'][k, l, :]

def print_Z():
    Z = []
    for i in range(data.shape[0]):
        z = trace['Z_{%d}'%i]
        Z.append(z)
    return np.array(Z)

def print_Pi_star():
    Pi_stars = []
    for i in range(data.shape[0]):
        Pi_star = trace['Pi_star_{%d}'%i]
        Pi_stars.append(Pi_star)
    return np.array(Pi_stars)

with open('./data/trace.pkl', 'rb') as t:
    pickle.dump(trace, t)
