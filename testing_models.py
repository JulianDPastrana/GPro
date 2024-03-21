import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data
from metrics import negatve_log_predictive_density, train_model, plot_gp_predictions, mean_squared_error


# Get train and test datasets
train_data, test_data = get_uv_data()
X_train, Y_train = test_data
X_test, Y_test = test_data

n_samples, input_dim = X_test.shape
observation_dim = Y_test.shape[1]

# Determine dimensions for the latent variables and inducing points
latent_dim = 2 * observation_dim
ind_process_dim = 10  # Number of independent processes in the coregionalization model

# Set up some common model's inducing hyperparameters
M = 50 # Number of inducing points
Zinit = np.random.rand(M, input_dim)

# Independent Multi-Output Gaussian Process with Gaussian Likelihood
kern_list = [
    gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(observation_dim)
]
kernel = gpf.kernels.SeparateIndependent(kern_list)
Zs = [Zinit.copy() for _ in range(observation_dim)]
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
model_ind = gpf.models.SVGP(
    kernel=kernel,
    likelihood=gpf.likelihoods.Gaussian(),
    inducing_variable=iv,
    num_latent_gps=observation_dim
)

# LMC Multi-Output Gaussian Process with Gaussian Likelihood
kern_list = [
    gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(ind_process_dim)
]
kernel = gpf.kernels.LinearCoregionalization(
    kern_list, W=np.random.randn(observation_dim, ind_process_dim)
)
Zs = [Zinit.copy() for _ in range(ind_process_dim)]
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
q_mu = np.zeros((M, ind_process_dim))
q_sqrt = np.repeat(np.eye(M)[None, ...], ind_process_dim, axis=0) * 1.0
model_lmc = gpf.models.SVGP(
    kernel=kernel,
    likelihood=gpf.likelihoods.Gaussian(),
    inducing_variable=iv,
    q_mu=q_mu,
    q_sqrt=q_sqrt
)

# Homogeneous Chained Multi-Output Gaussian Process with LogNormal Likelihood and Independent
Zs = [Zinit.copy() for _ in range(latent_dim)]
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
kern_list = [
    gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(latent_dim)
]
kernel = gpf.kernels.SeparateIndependent(kern_list)
model_hc_ind = gpf.models.SVGP(
    kernel=kernel,
    likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
    inducing_variable=iv,
    num_latent_gps=latent_dim
)

# Homogeneous Chained Multi-Output Gaussian Process with LogNormal Likelihood and Correlated
kern_list = [
    gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(ind_process_dim)
]
kernel = gpf.kernels.LinearCoregionalization(
    kern_list, W=np.random.randn(latent_dim, ind_process_dim)
)
Zs = [Zinit.copy() for _ in range(ind_process_dim)]
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
q_mu = np.zeros((M, ind_process_dim))
q_sqrt = np.repeat(np.eye(M)[None, ...], ind_process_dim, axis=0) * 1.0
model_hc_cor = gpf.models.SVGP(
    kernel=kernel,
    likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
    inducing_variable=iv,
    q_mu=q_mu,
    q_sqrt=q_sqrt
)

# Training and Testing the Models
for model, name in zip([model_ind, model_lmc, model_hc_ind, model_hc_cor], ["Independent", "LMC", "Chained_Ind", "Chainde_Corr"]):
    train_model(model, train_data, epochs=500, verbose=False)
    nlogpred = negatve_log_predictive_density(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    print(f"{name} - NLPD: {nlogpred.numpy():.2e}, MSE: {mse.numpy():.2e}")
    plot_gp_predictions(model, X_test, Y_test, name)