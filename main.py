import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data
from gpflow.utilities import print_summary

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
        kern_list, W=np.random.randn(latent_dim, ind_process_dim)
    )
    
    # Logging for debugging and verification purposes
    print("Observation dim:", likelihood.observation_dim)
    print("Latent dim:", likelihood.latent_dim)

    # Number of inducing points
    M = 250
    
    Z = np.random.rand(M, input_dim)
    # initialization of inducing input locations, one set of locations per output
    Zs = [Z.copy() for _ in range(ind_process_dim)]
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

def train_model(model, data, epochs=100, log_freq=20):
    """
    Trains the model for a specified number of epochs.
    """
    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.01)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.01)


    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    for epoch in range(1, epochs + 1):
        optimisation_step()

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % log_freq == 0 and epoch > 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")


def main():
    train_data, test_data = get_uv_data()
    X_train, Y_train = test_data
    X_test, Y_test = test_data

    model = build_model(train_data)
    print_summary(model)

    train_model(model, train_data, epochs=500)
    
    Y_mean, Y_var = model.predict_y(X_test)
    X_range = range(X_test.shape[0])

    observation_dim = Y_test.shape[1]
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

    # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.show()

    

if __name__ == "__main__": 
    main()