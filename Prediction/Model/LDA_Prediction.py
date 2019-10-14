import numpy as np
import scipy 
from scipy.special import gammaln
from scipy.stats import beta
from scipy.stats import dirichlet
from scipy.stats import multinomial
from tqdm import tqdm
import random
        
'''
save logs
'''
f = open('LDA_log', 'w')

'''
Utility Functions
'''
def get_pi(V, K):
    Pi = np.zeros(K)
    for k in range(K):
        if k == 0:
            Pi[k] = V[k]
        else:
            Pi[k] = V[k] * np.prod(1-V[:k])
    return Pi


# Updating Functions
def P_v_k(alpha, Z, k):
    hyper1_k = 1 + np.sum(Z, axis=0)[k]
    hyper2_k = alpha + np.sum(np.sum(Z, axis=0)[k+1:])
    v_k = np.random.beta(hyper1_k, hyper2_k)
    return v_k

def P_Pi(alpha, Z):
    hyper = alpha + np.sum(Z, axis=0)
    Pi = np.random.dirichlet(hyper)
    return Pi


def P_mu_klt(beta1, beta2, Z, k, l, t, X):
    hyper1_klt = beta1 + np.sum(X[:, l, t] * Z[:, k])
    hyper2_klt = beta2 + np.sum((1 - X[:, l, t]) * Z[:, k])
    mu_klt = np.random.beta(hyper1_klt, hyper2_klt)
    return mu_klt

def P_theta_kw(gamma, Q, Z, k):
    hyper = gamma + np.sum(Q[:, :] * np.expand_dims(Z[:, k], axis=-1), axis=0) # (1, W)
    theta_kw  = np.random.dirichlet(hyper)
    # print(theta_kw)
    return theta_kw

def P_Pi_star(n, Pi, M, Q, X, theta):
    condp_qn_theta_K = np.ones(Pi.shape[0])
    condp_xn_mu_K = np.zeros(Pi.shape[0])
    Pi_star = np.zeros(Pi.shape[0])

    for k in range(Pi.shape[0]):
        #Calculate condp_qn_theta
        multinom = scipy.stats.multinomial(1, theta[k])
        condp_qn_theta_K[k] = multinom.pmf(Q[n,:])
        #Calculate condp_xn_muk
        condp_xn_mu_K[k] = np.prod(np.power(M[k, :, :], X[n, :, :]) * np.power(1 - M[k, :, :], 1 - X[n, :, :])) #scalar
    # print(condp_xn_mu_K)
    denom = np.sum(np.prod(np.prod(np.power(M[:, :, :], X[n, :, :]) * np.power(1 - M[:, :, :], 1 - X[n, :, :]), axis=-1), axis=-1) * Pi * condp_qn_theta_K)
    for k in range(Pi.shape[0]):
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
            #print('We are in the [%d] iteration'%iter)
            for k in range(self.K):
                #Calculate v_k for all day types
                # self.V[k] = P_v_k(self.alpha, self.Z, k)
                #print("Iter in day %d"%day)
                #Calculate mu_klt for all day types, appliances and time
                for l in range(self.L):
                    for t in range(self.T):
                        self.M[k, l, t] = P_mu_klt(self.beta1, self.beta2, self.Z, k, l, t, self.X)

                #Calculate theta_kw for all day types and week of day
                # for w in range(self.W):
                self.theta[k, :] = P_theta_kw(self.gamma, self.Q, self.Z, k)
            Pi = P_Pi(self.alpha, self.Z)
            #print(self.theta)
            # print("Finish updating V, M, theta")
            # for k in range(self.M.shape[0]):
            #     for l in range(self.M.shape[1]):
            #         print(self.M[k, l, :])
            # print('V value is: ', self.V)
            """
            Updating Z
            """
            # Pi = get_pi(self.V, self.K)
            # print('Pi value is: ', Pi
            if iter > 0 and iter%1000 == 0:
                f.write('-----------------------------------------------\n')
                f.write('We are in [%d] iters\n'%iter)

            avg_likelihood = 0
            Pi_star = np.zeros_like(Pi)
            for day in range(self.X.shape[0]):
                Pi_star = P_Pi_star(day, Pi, self.M, self.Q, self.X, self.theta)
                self.Z[day, :] = np.random.multinomial(1, Pi_star)
                # print('Pi_star value is: ', Pi_star)
                likelihood_n = np.log(np.prod(np.power(np.power(self.M, self.X[day, :, :]) * np.power(1 - self.M, 1 - self.X[day, :, :]), np.expand_dims(np.expand_dims(self.Z[day, :], axis=-1), axis=-1))))
                if iter > 0 and iter%1000 == 0:
                    # print(Pi_star)
                    f.write('We assign [%d] day type to day %d with probability %.4f and its log-likelihood is %.4f.\n'%(np.where(self.Z[day, :]==1)[0], day, Pi_star[np.where(self.Z[day, :])], likelihood_n))
                avg_likelihood += likelihood_n
            if iter > 0 and iter%1000 == 0:    
                f.write('The average likelihood is: %.4f.\n'%(avg_likelihood/31))
                f.write('-----------------------------------------------\n')
                f.write('\n')
            """
            Print likelihood
            """
            # p_likelihood = 0
            # # for n in range(self.Z.shape[0]):
            # n = 0
            # p_likelihood += np.log(np.prod(np.power(np.power(self.M, self.X[n, :, :]) * np.power(1 - self.M, 1 - self.X[n, :, :]), np.expand_dims(np.expand_dims(self.Z[n, :], axis=-1), axis=-1))))
            # print('Log likelihood is %.4f'%(-p_likelihood/self.Z.shape[0]))

            # if self.verbose:
            #     t.set_postfix(NegLikelihood=p_likelihood)


    def infer(self, n):
        w = None
        w_truth = np.where(self.Q[n, :] == 1)[0]

        # Break when we get the right day type
        Pi = P_Pi(self.alpha, self.Z)
        # print(Pi)
        Pi_str = np.array2string(Pi)
        f.write(Pi_str)
        f.write('\n')
        while w != w_truth:
            self.Z[n, :] = np.random.multinomial(1, Pi)
            z_n = np.where(self.Z[n, :] == 1)[0]
            w = np.random.multinomial(1, np.squeeze(self.theta[z_n, :], axis=0))
            w = np.where(w == 1)[0]


        # Get day type pattern
        pattern_n = self.M[z_n, 0, :]
        # print('Our inferred pattern is {}'.format(pattern_n))
        # print('Real usage is {}'.format(self.X[n, :]))
        f.write('Our inferred pattern is {}\n'.format(pattern_n))
        f.write('Real usage is {}\n'.format(self.X[n, :]))




    def finish_writing(self):
        f.close()



