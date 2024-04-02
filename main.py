import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
import seaborn as sns
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data
from gpflow.utilities import print_summary
from metrics import negatve_log_predictive_density, train_model

def build_model(train_data):
    """
    Builds and returns a Gaussian Process (GP) model using the SVGP (Sparse Variational Gaussian Process) framework.

    Parameters:
    - train_data: tuple of (X, Y) where `X` is the training inputs and `Y` is the training outputs.

    Returns:
    - model: The constructed GP model configured with a specified kernel, likelihood, and inducing points.
    """
    # Unpack training data
    X, Y = train_data
    n_samples, input_dim = X.shape
    observation_dim = Y.shape[1]
    
    # Determine dimensions for the latent variables and inducing points
    latent_dim = 2 * observation_dim
    ind_process_dim = 10  # Number of independent processes in the coregionalization model

    # Initialize the likelihood with appropriate dimensions
    likelihood = LogNormalLikelihood(input_dim, latent_dim, observation_dim)
    
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=[1 for i in range(input_dim)]) for _ in range(ind_process_dim)]
    
    # Initialize the mixing matrix for the coregionalization kernel
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.normal(size=(latent_dim, ind_process_dim), scale=0.01)
    )
    
    # Logging for debugging and verification purposes
    print("Observation dim:", likelihood.observation_dim)
    print("Latent dim:", likelihood.latent_dim)

    # Number of inducing points
    M = 50
    
    Zinit = np.random.rand(M, input_dim)
    # initialization of inducing input locations, one set of locations per output
    Zs = [Zinit.copy() for _ in range(ind_process_dim)]
    # initialize as list inducing inducing variables
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    # create multi-output inducing variables from iv_list
    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    # Configure the inducing variables for the model
    # inducing_variable = gpf.inducing_variables.SharedIndependentInducingVariables(
    #     gpf.inducing_variables.InducingPoints(Z)
    # )

    # Initialize the mean and variance of the variational posterior
    q_mu = np.zeros((M, ind_process_dim))
    q_sqrt = np.repeat(np.eye(M)[None, ...], ind_process_dim, axis=0) * 1.0

    # Construct the SVGP model with the specified components
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        # num_latent_gps=ind_process_dim,  # Correctly set to match the latent dimensionality
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )
    
    return model


def main():
    train_data, val_data, test_data = get_uv_data()
    X_test, Y_test = test_data

    model = build_model(train_data)

    train_model(model, train_data, validation_data=val_data, epochs=500)
    print_summary(model)

    Y_mean, Y_var = model.predict_y(X_test)
    X_range = range(X_test.shape[0])

    observation_dim = Y_test.shape[1]
    y_lower = Y_mean - 1.96 * np.sqrt(Y_var)
    y_upper = Y_mean + 1.96 * np.sqrt(Y_var)

    # Calculate the number of rows and columns for the subplot matrix
    n_cols = int(np.ceil(np.sqrt(observation_dim)))
    n_rows = int(np.ceil(observation_dim / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()

    for d in range(observation_dim):
        ax_flat[d].scatter(X_range, Y_test[:, d], c="k", label="Test Data")
        ax_flat[d].plot(X_range, Y_mean[:, d], label="Predicted Mean")
        ax_flat[d].legend()
        ax_flat[d].fill_between(
        X_range, y_lower[:, d], y_upper[:, d], color="C0", alpha=0.1, label="CI"
        )


    # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.savefig("predictions.png")
    plt.close()

    nlogpred = negatve_log_predictive_density(model, X_test, Y_test)
    print(nlogpred)
    

if __name__ == "__main__": 
    main()