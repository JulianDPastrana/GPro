import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import pandas as pd
from likelihoods import MOChainedLikelihoodMC
from data_exploration import get_daily_vol_data
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import tensorflow_probability as tfp
from metrics import *

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

def chained_corr(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing, X_train):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(ind_process_dim)]
        
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.eye(latent_dim, ind_process_dim) + 1e-3
    )
    
    Zinit = X_train[np.random.choice(X_train.shape[0], num_inducing, replace=False), :]
    Zs = [Zinit.copy() for _ in range(ind_process_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    
    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    
    likelihood = likelihood = MOChainedLikelihoodMC(
            input_dim=input_dim,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            distribution_class=tfp.distributions.Normal,
            param1_transform=lambda x: x,
            param2_transform=tf.math.softplus
        )
    
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    return model


def lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(ind_process_dim)]
        
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.eye(observation_dim, ind_process_dim)+1e-3
    )
    
    Zinit = X_train[np.random.choice(X_train.shape[0], num_inducing, replace=False), :]
    Zs = [Zinit.copy() for _ in range(ind_process_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    
    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    
    likelihood = gpf.likelihoods.Gaussian(variance=[1]*observation_dim)

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    return model

def ind_gp(input_dim, observation_dim, num_inducing, X_train):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(observation_dim)]
        
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    
    Zinit = X_train[np.random.choice(X_train.shape[0], num_inducing, replace=False), :]
    Zs = [Zinit.copy() for _ in range(observation_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)

    
    likelihood = gpf.likelihoods.Gaussian(variance=[1]*observation_dim)

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        num_latent_gps=observation_dim
    )

    return model

def main():
    # Set seed for NumPy
    np.random.seed(0)

    # Set seed for GPflow
    tf.random.set_seed(0)

    # Define the grid of parameters to search
    order_values = [1, 2]#[1, 2, 3, 7, 14, 30]
    num_inducing_values = [8, 16]#[2 ** i for i in range(3, 10)]
    ind_process_dim_values = [32, 50]#[2 ** i for i in range(3, 10)]

    n_splits = 5
    # Initialize TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=500, test_size=500)
    # Store results
    results = []

    for order, num_inducing, ind_process_dim in product(order_values, num_inducing_values, ind_process_dim_values):
        print(f"Training with order={order} num_inducing={num_inducing}, ind_process_dim={ind_process_dim}")

        train_data, _ = get_daily_vol_data(input_width=order, label_width=1, shift=1)
        X, Y = train_data

        _, input_dim = X.shape
        observation_dim = Y.shape[1]
        latent_dim = 2 * observation_dim

        # Store metrics for all folds
        fold_metrics = {'NLPD': [], 'MSE': [], 'MAE': []}

        for train_index, test_index in tscv.split(X):
        
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            print(f"\t Train shape={X_train.shape}, Test shape={X_test.shape}")

            model = chained_corr(
                input_dim,
                latent_dim,
                observation_dim,
                ind_process_dim,
                num_inducing,
                X_train
            )

            train_model(model, (X_train, Y_train), batch_size=64, epochs=50)

            nlpd = negative_log_predictive_density(model, X_test, Y_test)
            msll = mean_standardized_log_loss()
            mse = mean_squared_error(model, X_test, Y_test)
            mae = mean_absolute_error(model, X_test, Y_test)

            fold_metrics['NLPD'].append(nlpd.numpy())
            fold_metrics['MSE'].append(mse.numpy())
            fold_metrics['MAE'].append(mae.numpy())
            print(f"\tNLPD: {nlpd.numpy():.2e}, MSE: {mse.numpy():.2e}, MAE: {mae.numpy():.2e}")

        # Calculate mean and variance for each metric across folds
        summary = {
            'order': order,
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
    filename = "chdgp_normal_gsresults.xlsx"
    df_results.to_excel("./results/gp_results"+filename, index=False)

    print(f"Grid search with time-series cross-validation completed. Results saved to {filename}.")

if __name__ == "__main__": 
    main()
