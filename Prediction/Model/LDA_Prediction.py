import numpy as np
import scipy as sp
from scipy.special import gammaln
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.stats import multinomial
from tqdm import tqdm
import random

'''
Utility Functions
'''
def get_pi(V):
    V_hat = 1 - np.repeat(np.expand_dims(V, axis=0), [K], axis=0) #This is a K-by-K matrix
    np.fill_diagonal(V_hat, 1) #Fill diagonal with ones
    V_hat = np.prod(V_hat, axis=1) #This is a (K,) matrix
    Pi = V * V_hat


# Updating Functions
def P_v_k(alpha, Z, k):
    hyper1_k = 1 + np.sum(Z, axis=0)[k]
    hyper2_k = alpha + np.sum(np.sum(Z, axis=0)[k+1:])
    v_k = np.random.beta(hyper1_k, hyper2_k)
    return v_k

def P_mu_klt(beta1, beta2, Z, k, l, t, X):
    hyper1_klt = beta1 + np.sum(X[:, l, t] * Z[:, k])
    hyper2_klt = beta2 + np.sum((1 - X[:, l, t]) * Z[:, k])
    mu_klt = np.random.beta(hyper1_klt, hyper2_klt)
    return mu_klt

def P_theta_kw(gamma, Q, Z, k):
    hyper = gamma + np.sum(Q[:, k, :] * Z, axis=0) # (1, W)
    theta_kw  = np.random.dirichlet(hyper)
    return theta_kw

def P_z_n(Z, k, n, Pi, M, X):
    condp_xn_muk = np.prod(np.power(M[k, :, :], X[n, :, :]) * np.power(1 - M[k, :, :], 1 - X[n, :, :])) #scalar
    Pi_k = Pi[k] #scalar
    condp_qn_theta =  dirichlet.pdf(Q[n, k, :], theta[k]) # scalar
    numer = Pi_k * condp_xn_muk * condp_qn_theta #scalar
    denom = np.sum(np.prod(np.prod(np.power(M[:, :, :], X[n, :, :]) * np.power(1 - M[:, :, :], 1 - X[n, :, :]), axis=-1), axis=-1) * Pi * dirichlet.pdf(Q[n, :, :], theta))
    Pi_star = numer / denom
    z_n = np.random.multinomial(Pi_star)
    return z_n



'''
Algorithm
'''

class algo(object):
    def __init__(self, alpha, gamma, beta1, beta2, K, X, verbose=False):
        # Set hyperparamteres
        self.alpha = alpha
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.X = X
        assert isinstance(X, np.ndarray), 'X should be numpy matrix'
        assert len(X.shape) == 3, 'X should be a 3-dimension matrix'
        self.N = X.shape[0]
        self.K = K
        self.T = X.shape[-1]
        self.W = 7
        self.L = X.shape[1]
        self.verbose = verbose
        self._initialize()


    def _initialize(self):
        # Parameters
        if self.verbose:
            print('initializing Parameters...')
        self.Z = np.zeros((self.N, self.K)) #K is the number of day types
        self.M = np.random.beta(self.beta1, self.beta2, size=(self.K, self.L, self.T))
        self.V = np.random.beta(1, self.alpha, size=self.K)
        self.theta = np.random.dirichlet(self.gamma, size=self.K)
        self.Q = np.zeros((self.N, self.K, self.W))
        for k in range(self.K):
            self.Q[:, k, :] = np.random.multinomial(1, self.theta[k], size=self.N)

        # Random initialize our matrices
        for day in range(self.X.shape[0]):
            for app in range(self.X.shape[1]):
                for hour in range(self.X.shape[2]):
                    z_n  = random.randint(0,self.K-1)
                    self.Z[day, z_n] += 1

    def train(self, iters=300):
        if self.verbose:
            t = tqdm(range(iters), desc='Gibbs Sampling:')
        else:
            t = range(iters)
        for iter in t:
            """
            Updating V, M, theta
            """
            for day in range(self.X.shape[0]):

                #Calculate v_k for all day types
                for k in range(self.K):
                    self.V[k] = P_v_k(self.alpha, self.Z, k)

                    #Calculate mu_klt for all day types, appliances and time
                    for l in range(self.L):
                        for t in range(self.T):
                            self.M[k, l, t] = P_mu_klt(self.beta1, self.beta2, self.Z, k, l, t, self.X)

                    #Calculate theta_kw for all day types and week of day
                    for w in range(self.W):
                        self.theta[k, w] = P_theta_kw(self.gamma, self.Q, self.Z, k)

            """
            Updating Z
            """
            Pi = get_pi(self.V)
            self.Q = np.random.multinomial(1, self.theta, size=self.N)
            for day in self.X.shape[0]:
                for k in range(self.K):
                    #Posterior Z_n
                    self.Z[day, k] = P_z_n(self.Z, k, day, Pi, self.M, self.X)

            """
            Print likelihood
            """
            p_likelihood = -np.log(np.prod(np.power((np.power(self.M, self.X) * np.power(1 - self.M, 1 - self.X)), Z)))
            if self.verbose:
                t.set_postfix(NegLikelihood=p_likelihood)




