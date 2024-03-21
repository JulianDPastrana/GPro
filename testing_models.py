import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data


# Get train and test datasets
train_data, test_data = get_uv_data()
X_train, Y_train = test_data
X_test, Y_test = test_data

# Set up some common model's inducing hyperparameters
# Number of inducing points
M = 50
Z = np.random.rand(M, input_dim)
# initialization of inducing input locations, one set of locations per output
Zs = [Z.copy() for _ in range(ind_process_dim)]
# initialize as list inducing inducing variables
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
# create multi-output inducing variables from iv_list
inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
# Initialize the mean and variance of the variational posterior
q_mu = np.zeros((M, ind_process_dim))
q_sqrt = np.repeat(np.eye(M)[None, ...], ind_process_dim, axis=0) * 1.0