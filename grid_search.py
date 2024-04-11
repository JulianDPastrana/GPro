import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import pandas as pd
from likelihoods import MOChainedLikelihoodMC
from data_exploration import get_uv_data
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import tensorflow_probability as tfp
from metrics import *

def chained_corr(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) + gpf.kernels.Linear(variance=np.ones(input_dim)) + gpf.kernels.Constant() for _ in range(ind_process_dim)]
    
    # Initialize the mixing matrix for the coregionalization kernel
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.randn(latent_dim, ind_process_dim)
    )
    Zinit = np.random.rand(num_inducing, input_dim)
    Zs = [Zinit.copy() for _ in range(ind_process_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    likelihood = likelihood = MOChainedLikelihoodMC(
        input_dim=input_dim,
        latent_dim=latent_dim,
        observation_dim=observation_dim,
        distribution_class=tfp.distributions.Gamma,
        param1_transform=tfp.bijectors.Softplus(),
        param2_transform=tfp.bijectors.Softplus()
    )
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    return model


# Set seed for NumPy
np.random.seed(0)

# Set seed for GPflow
tf.random.set_seed(0)

(train_data, _, test_data), scaler = get_uv_data()
X_test, Y_test = test_data
X_train, Y_train = train_data


n_samples, input_dim = X_train.shape
observation_dim = Y_train.shape[1]

# Determine dimensions for the latent variables and inducing points
latent_dim = 2 * observation_dim
# Define the grid of parameters to search
num_inducing_values = [2 ** i for i in range(8)]
ind_process_dim_values = [2 ** i for i in range(8)]
n_splits = 5  # Number of splits for TimeSeriesSplit

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=1200, test_size=30)
# print(np.sum(np.isnan(Y_train)))
# Store results
results = []

# Grid search with time-series cross-validation
for num_inducing, ind_process_dim in product(num_inducing_values, ind_process_dim_values):
    print(f"Training with num_inducing={num_inducing}, ind_process_dim={ind_process_dim}")

    # Store metrics for all folds
    fold_metrics = {'NLPD': [], 'MSE': [], 'MAE': []}

    for train_index, test_index in tscv.split(X_train):
        print(f"\t Train shape={train_index.shape}, Test shape={test_index.shape}")
        X_train_fold, X_test_fold = X_train[train_index], X_train[test_index]
        Y_train_fold, Y_test_fold = Y_train[train_index], Y_train[test_index]

        model = chained_corr(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing)
        train_model(model, (X_train_fold, Y_train_fold), validation_data=(X_test_fold, Y_test_fold),
                    epochs=1000, verbose=False, patience=100)

        nlogpred = negatve_log_predictive_density(model, X_test_fold, Y_test_fold)
        mse = mean_squared_error(model, X_test_fold, Y_test_fold)
        mae = mean_absolute_error(model, X_test_fold, Y_test_fold)
        fold_metrics['NLPD'].append(nlogpred.numpy())
        fold_metrics['MSE'].append(mse.numpy())
        fold_metrics['MAE'].append(mae.numpy())
        print(f"\tNLPD: {nlogpred.numpy():.2e}, MSE: {mse.numpy():.2e}, MAE: {mae.numpy():.2e}")

    # Calculate mean and variance for each metric across folds
    summary = {
        'num_inducing': num_inducing,
        'ind_process_dim': ind_process_dim,
        'NLPD_mean': np.mean(fold_metrics['NLPD']),
        'NLPD_variance': np.var(fold_metrics['NLPD']),
        'MSE_mean': np.mean(fold_metrics['MSE']),
        'MSE_variance': np.var(fold_metrics['MSE']),
        'MAE_mean': np.mean(fold_metrics['MAE']),
        'MAE_variance': np.var(fold_metrics['MAE']),
    }

    results.append(summary)

# Convert results to DataFrame
df_results = pd.DataFrame(results)

# Save results to Excel file
filename = "mochainedgp_model_ts_grid_search_results.xlsx"
df_results.to_excel(filename, index=False)

print(f"Grid search with time-series cross-validation completed. Results saved to {filename}.")
