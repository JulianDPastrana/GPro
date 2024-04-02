import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import pandas as pd
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
from metrics import *

def chained_corr(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing):
    kern_list = [
        gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(ind_process_dim)
    ]
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.normal(size=(latent_dim, ind_process_dim), scale=0.01)+np.eye(latent_dim, ind_process_dim)
    )
    Zinit = np.random.rand(num_inducing, input_dim)
    Zs = [Zinit.copy() for _ in range(latent_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    # q_mu = np.zeros((num_inducing, ind_process_dim))
    # q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    # model_hc_cor = gpf.models.SVGP(
    #     kernel=kernel,
    #     likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
    #     inducing_variable=iv,
    #     q_mu=q_mu,
    #     q_sqrt=q_sqrt
    # )
    kern_list = [
    gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(latent_dim)
]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
        inducing_variable=iv,
        num_latent_gps=latent_dim
    )
    return model


# Get train and test datasets
train_data, val_data, test_data = get_uv_data()
X_train, Y_train = test_data


n_samples, input_dim = X_train.shape
observation_dim = Y_train.shape[1]

# Determine dimensions for the latent variables and inducing points
latent_dim = 2 * observation_dim
# Define the grid of parameters to search
num_inducing_values = [20, 50, 100]
ind_process_dim_values = [2, 5, 10]
n_splits = 5  # Number of splits for TimeSeriesSplit

# Initialize TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=n_splits)
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
        train_model(model, (X_train_fold, Y_train_fold), validation_data=(X_test_fold, Y_test_fold), epochs=500, verbose=False)

        nlogpred = negatve_log_predictive_density(model, X_test_fold, Y_test_fold)
        mse = mean_squared_error(model, X_test_fold, Y_test_fold)
        mae = mean_absolute_error(model, X_test_fold, Y_test_fold)
        fold_metrics['NLPD'].append(nlogpred.numpy())
        fold_metrics['MSE'].append(mse.numpy())
        fold_metrics['MAE'].append(mae.numpy())
        print(f"NLPD: {nlogpred.numpy():.2e}, MSE: {mse.numpy():.2e}, MAE: {mae.numpy():.2e}")

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
filename = "gp_model_ts_grid_search_results.xlsx"
df_results.to_excel(filename, index=False)

print(f"Grid search with time-series cross-validation completed. Results saved to {filename}.")
