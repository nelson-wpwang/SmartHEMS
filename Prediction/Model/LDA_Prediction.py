import numpy as np
import scipy 
from scipy.special import gammaln
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.stats import multinomial
from tqdm import tqdm
import random

'''
Utility Functions
'''
def get_pi(V, K):
    V_hat = 1 - np.repeat(np.expand_dims(V, axis=0), [K], axis=0) #This is a K-by-K matrix
    np.fill_diagonal(V_hat, 1) #Fill diagonal with ones
    V_hat = np.prod(V_hat, axis=1) #This is a (K,) matrix
    Pi = V * V_hat
    return Pi


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
    hyper = gamma + np.sum(Q[:, :] * np.expand_dims(Z[:, k], axis=-1), axis=0) # (1, W)
    theta_kw  = np.random.dirichlet(hyper)
    return theta_kw

def P_Pi_star(Z, n, Pi, M, Q, X, theta):
    condp_qn_theta_K = np.ones(Z.shape[1])
    condp_xn_mu_K = np.zeros(Z.shape[1])
    Pi_star = np.zeros(Z.shape[1])

    for k in range(Z.shape[1]):
        #Calculate condp_qn_theta
        multinom = scipy.stats.multinomial(1, theta[k])
        condp_qn_theta_K[k] = multinom.pmf(Q[n,:])
        #Calculate condp_xn_muk
        condp_xn_mu_K[k] = np.prod(np.power(M[k, :, :], X[n, :, :]) * np.power(1 - M[k, :, :], 1 - X[n, :, :])) #scalar
    # print(condp_xn_mu_K)
    denom = np.sum(np.prod(np.prod(np.power(M[:, :, :], X[n, :, :]) * np.power(1 - M[:, :, :], 1 - X[n, :, :]), axis=-1), axis=-1) * Pi * condp_qn_theta_K)
    for k in range(Z.shape[1]):
        numer = Pi[k] * condp_xn_mu_K[k] * condp_qn_theta_K[k] #scalar
        Pi_star[k] = numer / denom
    # print(Pi_star)
    return Pi_star



'''
Algorithm
'''

class algo(object):
    def __init__(self, alpha, gamma, beta1, beta2, K, X, Q, verbose=False):
        # Set hyperparamteres
        self.alpha = alpha
        self.gamma = np.array(gamma)
        self.beta1 = beta1
        self.beta2 = beta2
        self.X = X
        self.Q = Q
        assert isinstance(X, np.ndarray) and isinstance(Q, np.ndarray), 'X, Q should be numpy matrix'
        assert len(X.shape) == 3, 'X should be a 3-dimension matrix'
        assert len(Q.shape) == 2, 'Q should be a 2-dimension matrix'
        self.N = X.shape[0]
        self.K = K
        self.T = X.shape[-1]
        self.W = Q.shape[-1]
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
        

        # Random initialize our matrices
        for day in range(self.X.shape[0]):
            z_n  = random.randint(0,self.K-1)
            self.Z[day, z_n] += 1

    def train(self, iters=60):
        if self.verbose:
            t = tqdm(range(iters), desc='Gibbs Sampling:')
        else:
            t = range(iters)
        for iter in t:
            """
            Updating V, M, theta
            """
            print('We are in the [%d] iteration'%iter)
            for k in range(self.K):
                #Calculate v_k for all day types
                self.V[k] = P_v_k(self.alpha, self.Z, k)
                #print("Iter in day %d"%day)
                #Calculate mu_klt for all day types, appliances and time
                for l in range(self.L):
                    for t in range(self.T):
                        self.M[k, l, t] = P_mu_klt(self.beta1, self.beta2, self.Z, k, l, t, self.X)

                #Calculate theta_kw for all day types and week of day
                # for w in range(self.W):
                self.theta[k, :] = P_theta_kw(self.gamma, self.Q, self.Z, k)
            #print(self.theta)
            print("Finish updating V, M, theta")

            """
            Updating Z
            """
            Pi = get_pi(self.V, self.K)
            # print(Pi)
            Pi_star = np.zeros_like(Pi)
            for day in range(self.X.shape[0]):
                Pi_star = P_Pi_star(self.Z, day, Pi, self.M, self.Q, self.X, self.theta)
                self.Z[day, :] = np.random.multinomial(1, Pi_star)



            """
            Print likelihood
            """
            p_likelihood = 0
            for n in range(self.Z.shape[0]):
                p_likelihood += np.log(np.prod(np.power(np.power(self.M, self.X[n, :, :]) * np.power(1 - self.M, 1 - self.X[n, :, :]), np.expand_dims(np.expand_dims(self.Z[n, :], axis=-1), axis=-1))))
            print('Log likelihood is %.4f'%(-p_likelihood/self.Z.shape[0]))

            if self.verbose:
                t.set_postfix(NegLikelihood=p_likelihood)




