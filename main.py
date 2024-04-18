import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
import seaborn as sns
import pickle
from likelihoods import MOChainedLikelihoodMC, MOChainedLikelihoodQuad
from data_exploration import get_uv_data
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
from metrics import negatve_log_predictive_density, train_model

gpf.config.set_default_float(np.float64)

def chained_corr(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) + gpf.kernels.Linear(variance=np.ones(input_dim)) + gpf.kernels.Constant() for _ in range(ind_process_dim)]
    
    # Initialize the mixing matrix for the coregionalization kernel
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.randn(latent_dim, ind_process_dim) + np.eye(latent_dim, ind_process_dim)
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
        distribution_class=tfp.distributions.Beta,
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


def chained_indp(input_dim, latent_dim, observation_dim, num_inducing):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) + gpf.kernels.Linear(variance=np.ones(input_dim)) + gpf.kernels.Constant() for _ in range(latent_dim)]
    
    kernel = kernel = gpf.kernels.SeparateIndependent(kern_list)
    Zinit = np.random.rand(num_inducing, input_dim)
    Zs = [Zinit.copy() for _ in range(latent_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    q_mu = np.zeros((num_inducing, latent_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], latent_dim, axis=0) * 1.0
    likelihood = likelihood = MOChainedLikelihoodMC(
        input_dim=input_dim,
        latent_dim=latent_dim,
        observation_dim=observation_dim,
        distribution_class=tfp.distributions.Beta,
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


def main():
    # Set seed for NumPy
    np.random.seed(0)

    # Set seed for GPflow
    tf.random.set_seed(0)

    (train_data, val_data, test_data), scaler = get_uv_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data


    n_samples, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]

    # Determine dimensions for the latent variables and inducing points
    latent_dim = 2 * observation_dim


    model = chained_corr(input_dim, latent_dim, observation_dim, num_inducing=32, ind_process_dim=32)
    # gpf.utilities.set_trainable(model.kernel.W, False)
    train_model(model, train_data, validation_data=val_data, epochs=5000, patience=100)
    # print_summary(model)

    Y_mean, Y_var = model.predict_y(X_test)
    # print(Y_var)
    X_range = range(X_test.shape[0])

    observation_dim = Y_test.shape[1]
    y_lower = Y_mean - 1.0 * np.sqrt(Y_var)
    y_upper = Y_mean + 1.0 * np.sqrt(Y_var)

    # Calculate the number of rows and columns for the subplot matrix
    n_cols = int(np.ceil(np.sqrt(observation_dim)))
    n_rows = int(np.ceil(observation_dim / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax
    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()

    for d in range(observation_dim):
        ax_flat[d].scatter(X_range, Y_test[:, d], c="k", label="Test Data")
        ax_flat[d].plot(X_range, Y_mean[:, d], label="Predicted Mean")
        # ax_flat[d].plot(X_range, np.sqrt(Y_var[:, d]), label="Predicted Std")
        ax_flat[d].fill_between(
        X_range, y_lower[:, d], y_upper[:, d], color="C0", alpha=0.1, label="+/- Std"
        )
        ax_flat[d].legend()
        ax_flat[d].set_ylabel("Useful Volume")
        ax_flat[d].set_xlabel("Days")
        # ax_flat[d].set_xticks([])
        # ax_flat[d].set_yticks([])
        ax_flat[d].grid(True)




    # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.savefig("predictions_test.png")
    plt.close()

    nlogpred = negatve_log_predictive_density(model, X_test, Y_test)
    print(nlogpred)

######################################3
    Y_mean, Y_var = model.predict_y(X_train)
    # print(Y_var)
    X_range = range(X_train.shape[0])

    observation_dim = Y_test.shape[1]
    y_lower = Y_mean - 1.0 * np.sqrt(Y_var)
    y_upper = Y_mean + 1.0 * np.sqrt(Y_var)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax

    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()

    for d in range(observation_dim):
        ax_flat[d].scatter(X_range, Y_train[:, d], c="k", label="Test Data")
        ax_flat[d].plot(X_range, Y_mean[:, d], label="Predicted Mean")
        ax_flat[d].legend()
        ax_flat[d].fill_between(
        X_range, y_lower[:, d], y_upper[:, d], color="C0", alpha=0.1, label="CI"
        )


    # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.savefig("predictions_train.png")
    plt.close()

    nlogpred = negatve_log_predictive_density(model, X_test, Y_test)
    print(nlogpred)
    model_name = "Beta_model"
    with open(model_name, 'wb') as file:
        pickle.dump(gpf.utilities.parameter_dict(model), file)

    # plot_results(model, X_test, Y_test)

    

if __name__ == "__main__": 
    main()