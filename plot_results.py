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
from main import build_model
import pickle
from matplotlib import cm
from scipy.stats import gamma
from metrics import negatve_log_predictive_density, train_model
from typing import Any, Tuple


def plot_results(model: Any, X: np.ndarray, Y: np.ndarray, scaler: Any) -> None:
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
    num_samples = 200
    num_y_pixels = 100
    
    # Calculate X and Y limits for the plot
    xlim_min, xlim_max = time_index[:, 0].min(), time_index[:, 0].max()
    ylim_min, ylim_max = Y[:, 0].min(), Y[:, 0].max()
    
    # Sample from the model's predictive distribution
    Fsamples = model.predict_f_samples(X, num_samples)
    
    # Transform samples using the likelihood parameters
    alpha = model.likelihood.param1_transform(Fsamples[..., 0])
    beta = model.likelihood.param1_transform(Fsamples[..., 1])
    
    # Prepare line space for the Y-axis
    line_space = np.linspace(np.floor(ylim_min), np.ceil(ylim_max), num_y_pixels)
    predictive_density = np.zeros((X.shape[0], num_y_pixels))
    
    # Compute the predictive density
    for j in range(X.shape[0]):
        for i in range(num_samples):
            dist = model.likelihood.distribution_class(alpha[i, j], beta[i, j])
            predictive_density[j, :] += dist.prob(line_space).numpy()
    predictive_density /= num_samples
    
    # Inverse transform the line space and predictive density using the provided scaler
    line_space_transformed = scaler.inverse_transform(line_space[:, None]).flatten()
    predictive_density_transformed = scaler.inverse_transform(predictive_density)
    
    # Inverse transform the mean prediction and observations
    Ymean, _ = model.predict_y(X)
    Ymean_transformed = scaler.inverse_transform(Ymean)
    Y_transformed = scaler.inverse_transform(Y)
    
    # Create a meshgrid for the contour plot
    x_mesh, y_mesh = np.meshgrid(time_index, line_space_transformed)
    
    # Plot the predictive density and observations
    fig, ax = plt.subplots(figsize=(15, 8))
    cs = ax.contourf(x_mesh, y_mesh, predictive_density_transformed.T)
    fig.colorbar(cs, ax=ax)
    ax.scatter(time_index, Y_transformed, color="k", s=10, label="Observaciones")
    ax.plot(time_index, Ymean_transformed, color="white", lw=1, alpha=0.8, label="Media")
    ax.set_xlabel("Indice de tiempo del pronóstico [Días]")
    ax.set_ylabel("Generación térmica [Gwh]")
    ax.set_title("Densidad de probabilidad del pronóstico")
    plt.legend()

    plt.tight_layout()
    plt.savefig("predictive_density.png")
    plt.close()


def main():
    save_dir = "saved_model_0"
    model = tf.saved_model.load(save_dir)
    (train_data, val_data, test_data), scaler = get_uv_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data
    # Cargar el objeto desde un archivo
    with open('save_dir', 'rb') as file:
        params = pickle.load(file)
    model = build_model(train_data)
    gpf.utilities.multiple_assign(model, params)


    plot_results(model, X_test, Y_test, scaler)

if __name__ == "__main__": 
    main()