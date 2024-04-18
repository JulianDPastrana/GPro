import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
import seaborn as sns
from likelihoods import LogNormalLikelihood, LogNormalMCLikelihood, LogNormalQuadLikelihood, HeteroskedasticLikelihood
from data_exploration import get_uv_data
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
import matplotlib
from main import chained_corr
import pickle
from matplotlib import cm
from scipy.stats import gamma
from metrics import negatve_log_predictive_density, train_model
from typing import Any, Tuple


def plot_results(model: Any, X: np.ndarray, Y: np.ndarray) -> None:
    """
    Plot the predictive density of a model's forecasts along with the observed data.

    Parameters:
    model (Any): The trained model that provides predictive distribution and likelihood functions.
    X (np.ndarray): The input features of the dataset, typically time indices in forecast scenarios.
    Y (np.ndarray): The target values of the dataset, typically quantities like power generation.
    scaler (Any): A scaler object that provides inverse_transform functionality for normalization.

    The function saves the plot as 'predictive_density.png' in the current directory.

    The plotting includes the observation points, the mean of the forecast, and the
    contour fill of the predictive density distribution.
    """
    
    # Initialize the time index for X-axis
    time_index = np.arange(X.shape[0])[:, None]
    # Define constants for the number of samples and Y-axis pixels
    num_samples = 10
    num_y_pixels = 20
    observation_dim = Y.shape[1]
    # Calculate X and Y limits for the plot
    xlim_min, xlim_max = time_index[:, 0].min(), time_index[:, 0].max()
    ylim_min, ylim_max = 0, 1
    
    # Sample from the model's predictive distribution
    Fsamples = model.predict_f_samples(X, num_samples)
    
    # Transform samples using the likelihood parameters
    alpha = model.likelihood.param1_transform(Fsamples[..., ::2])
    beta = model.likelihood.param1_transform(Fsamples[..., 1::2])
    
    # Prepare line space for the Y-axis
    line_space = np.linspace(ylim_min, ylim_max, num_y_pixels)
    predictive_density = np.zeros((X.shape[0], num_y_pixels, observation_dim))
    
    # Compute the predictive density
    for j in range(X.shape[0]):
        for i in range(num_samples):
            for d in range(observation_dim):
                dist = model.likelihood.distribution_class(alpha[i, j, d], beta[i, j, d], force_probs_to_zero_outside_support=True)
                predictive_density[j, :, d] += dist.prob(line_space).numpy()
    predictive_density /= num_samples

    
    # Create a meshgrid for the contour plot
    x_mesh, y_mesh = np.meshgrid(time_index, line_space)
    n_cols = int(np.ceil(np.sqrt(observation_dim)))
    n_rows = int(np.ceil(observation_dim / n_cols))

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    ax = np.array([ax]) if not isinstance(ax, np.ndarray) else ax
    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()
    for d in range(observation_dim):
        cs = ax_flat[d].contourf(x_mesh, y_mesh, np.log10(predictive_density[:, :, d].T))
        # plt.colorbar()
        # fig.colorbar(cs, ax=ax)
        ax_flat[d].scatter(time_index, Y[:, d], color="k", s=10, label="Observaciones")
        # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.savefig("predictive_density.png")
    plt.close()

def main():
    save_dir = "Beta_model"
    # model = tf.saved_model.load(save_dir)
    (train_data, val_data, test_data), scaler = get_uv_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data
    # Cargar el objeto desde un archivo
    with open(save_dir, 'rb') as file:
        params = pickle.load(file)

    n_samples, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]

    # Determine dimensions for the latent variables and inducing points
    latent_dim = 2 * observation_dim


    model = chained_corr(input_dim, latent_dim, observation_dim, num_inducing=32, ind_process_dim=32)
    gpf.utilities.multiple_assign(model, params)


    plot_results(model, X_test, Y_test)

if __name__ == "__main__": 
    main()