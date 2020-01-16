import numpy as np
import pymc3 as pm
"""
Naive Bayes
"""
# Hyper-parameter for uniform Dirichlet prior
alpha = np.ones(K)

# Hyper-parameter for sparse Dirichlet prior
beta = np.ones(V) * 0.3

with pm.Model() as model:
    # Global topic distribution
    theta = pm.Dirichlet('theta', a=alpha)

    # Word distributions for K topics
    phi = pm.Dirichlet('phi', a=beta, shape=(K, V))

    # Topic of documents
    z = pm.Categorical('z', p=theta, shape=(D,))

    for i in range(D):
        # Words of document
        w = pm.Categorical(f'w_{i}', p=phi[z[i]], observed=X[i])

with model:
    trace = pm.sample(2000, chains=1)


# """
# LDA
# """
# alpha = np.ones(K) * 0.3
# beta = np.ones(V) * 0.3
#
# with pm.Model() as model:
#     phi = pm.Dirichlet("phi", a=beta, shape=(K, V))
#
#     for i in range(D):
#         theta = pm.Dirichlet(f"theta_{i}", a=alpha)
#
#         z = pm.Categorical(f"z_{i}", p=theta, shape=X[i].shape)
#         w = pm.Categorical(f"w_{i}", p=phi[z], observed=X[i])
#
# with model:
#     trace = pm.sample(2000, chains=1, nuts_kwargs={'target_accept': 0.9})
