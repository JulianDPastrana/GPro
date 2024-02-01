import sys
import climin
from functools import partial
import warnings
import os
sys.path.append('..')

import numpy as np
from scipy.stats import multinomial
from scipy.linalg.blas import dtrmm

import GPy
from GPy.util import choleskies
from GPy.core.parameterization.param import Param
from GPy.kern import Coregionalize
from GPy.likelihoods import Likelihood
from GPy.util import linalg

from likelihoods.bernoulli import Bernoulli
from likelihoods.gaussian import Gaussian
from likelihoods.categorical import Categorical
from likelihoods.hetgaussian import HetGaussian
from likelihoods.beta import Beta
from likelihoods.gamma import Gamma
from likelihoods.exponential import Exponential

from hetmogp.util import draw_mini_slices
from hetmogp.het_likelihood import HetLikelihood
from hetmogp.svmogp import SVMOGP
from hetmogp import util
from hetmogp.util import vem_algorithm as VEM

import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from matplotlib import rc, font_manager
from matplotlib import rcParams
# from matplotlib2tikz import save as tikz_save

warnings.filterwarnings("ignore")
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin'



M = 8  # number of inducing points
Q = 2  # number of latent functions

# Heterogeneous Likelihood Definition
likelihoods_list = [Gaussian(sigma=1.), Bernoulli()] # Real + Binary
likelihood = HetLikelihood(likelihoods_list)
Y_metadata = likelihood.generate_metadata()
D = likelihood.num_output_functions(Y_metadata)

W_list, _ = util.random_W_kappas(Q, D, rank=1, experiment=True)

X1 = np.sort(np.random.rand(600))[:, None]
X2 = np.sort(np.random.rand(500))[:, None]
X = [X1, X2]


# True U and F functions
def experiment_true_u_functions(X_list):
    u_functions = []
    for X in X_list:
        u_task = np.empty((X.shape[0],2))
        u_task[:,0,None] = 4.5*np.cos(2*np.pi*X + 1.5*np.pi) - \
                           3*np.sin(4.3*np.pi*X + 0.3*np.pi) + \
                           5*np.cos(7*np.pi * X + 2.4*np.pi)
                
        u_task[:,1,None] = 4.5*np.cos(1.5*np.pi*X + 0.5*np.pi) + \
                   5*np.sin(3*np.pi*X + 1.5*np.pi) - \
                   5.5*np.cos(8*np.pi * X + 0.25*np.pi)

        u_functions.append(u_task)
    return u_functions


def experiment_true_f_functions(true_u, X_list):
    true_f = []
    W = W_lincombination()
    
    # D=1
    for d in range(2):
        f_d = np.zeros((X_list[d].shape[0], 1))
        for q in range(2):
            f_d += W[q][d].T*true_u[d][:,q,None]
        true_f.append(f_d)

    return true_f

# True Combinations
def W_lincombination():
    W_list = []
    # q=1
    Wq1 = np.array(([[-0.5],[0.1]]))
    W_list.append(Wq1)
    # q=2
    Wq2 = np.array(([[-0.1],[.6]]))
    W_list.append(Wq2)
    return W_list

# True functions values for inputs X
trueU = experiment_true_u_functions(X)
trueF = experiment_true_f_functions(trueU, X)

# Generating training data Y (sampling from heterogeneous likelihood)
Y = likelihood.samples(F=trueF, Y_metadata=Y_metadata) 

# Plot true parameter functions PFs (black) and heterogeneous data (blue, orange)
plt.figure(figsize=(10, 6))
Ntask = 2
for t in range(Ntask):
    plt.plot(X[t],trueF[t],'k-', alpha=0.75)
    plt.plot(X[t],Y[t],'+')
    
plt.show()