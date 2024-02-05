import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import matplotlib.colors
import gpflow as gpf
from gpflow.likelihoods import MultiLatentTFPConditional
from check_shapes import check_shapes
from typing import Any, Callable
from matplotlib import cm, ticker
from datasets import streamflow_dataset

class HeteroskedasticLikelihood(MultiLatentTFPConditional):
    """
    Custom Likelihood class that extends MultiLatentTFPConditional to handle
    heteroskedastic scenarios.
    """
    def __init__(self, distribution_class, param1_transform, param2_transform, **kwargs: Any):
        self.param1_transform = param1_transform
        self.param2_transform = param2_transform

        @check_shapes("F: [batch..., 2]")
        def conditional_distribution(F) -> tfp.distributions.Distribution:
            param1 = self.param1_transform(F[..., :1])
            param2 = self.param2_transform(F[..., 1:])
            return distribution_class(param1, param2)

        super().__init__(latent_dim=2, conditional_distribution=conditional_distribution, **kwargs)

def initialize_data(N):
    """
    Initializes data for the model.
    """
    np.random.seed(0)
    tf.random.set_seed(0)

    X = np.linspace(0, 4 * np.pi, N)[:, None]
    f1 = np.sin(X)
    f2 = np.cos(X)

    # Ensure positive scale values
    alpha = f1
    beta = np.exp(f2)

    Y = pred_distribution(alpha, beta).sample().numpy()
    return X, Y

def build_model(X, Y):
    """
    Builds and returns the GP model.
    """
    likelihood = HeteroskedasticLikelihood(
        distribution_class=pred_distribution,
        param1_transform=tfp.bijectors.Exp(),
        param2_transform=tfp.bijectors.Exp(),
    )

    kernel = gpf.kernels.SeparateIndependent(
        [gpf.kernels.SquaredExponential(), gpf.kernels.SquaredExponential()]
    )

    M = 20
    Z = np.linspace(X.min(), X.max(), M)[:, None]

    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [gpf.inducing_variables.InducingPoints(Z), gpf.inducing_variables.InducingPoints(Z)]
    )

    model = gpf.models.SVGP(kernel=kernel, likelihood=likelihood, inducing_variable=inducing_variable,
                            num_latent_gps=likelihood.latent_dim)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    return model

def train_model(model, data, epochs=100, log_freq=20):
    """
    Trains the model for a specified number of epochs.
    """
    X, Y = data
    loss_fn = model.training_loss_closure(data)
    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.001)
    # adam_var = tf.optimizers.Adam(0.001)
    adam_opt = tf.optimizers.Adam(0.01)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    adam_vars = model.trainable_variables

    @tf.function
    def optimization_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    loss = []
    for epoch in range(1, epochs + 1):
        optimization_step()
        if epoch % log_freq == 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
            Ymean, _ = model.predict_y(X)
            Xrange = np.arange(X.shape[0])
            plt.plot(Xrange, Ymean, color="k")
            plt.scatter(Xrange, Y)
            plt.show()


def plot_results(model, X, Y):
    """
    Plots the results of the model.
    """
    Xrange = np.arange(X.shape[0])[:, None]
    num_samples = 100

    num_y_pixels = 100
    #Want the top left pixel to be evaluated at 1
    
    xlim_min, xlim_max = Xrange[:, 0].min(), Xrange[:, 0].max()
    ylim_min, ylim_max = Y[:, 0].min(), Y[:, 0].max()

    Fsamples = model.predict_f_samples(X, num_samples)
    print(Fsamples.shape)
    alpha = np.exp(Fsamples[..., 0].numpy().squeeze())
    beta = np.exp(Fsamples[..., 1].numpy().squeeze())
    line = np.linspace(np.floor(ylim_min), np.ceil(ylim_max), num_y_pixels)
    res = np.zeros((X.shape[0], num_y_pixels))
    for j in range(X.shape[0]):
        sf = alpha[:, j]  # Pick out the jth point along X axis
        sg = beta[:, j]
        for i in range(num_samples):
            # Pick out the sample and evaluate the pdf on a line between 0
            # and 1 with these alpha and beta values
            dist = pred_distribution(sf[i], sg[i])
            res[j, :] += dist.prob(line).numpy()#distribution.pdf(line, sf[i], sg[i])
        res[j, :] /= num_samples

    vmax, vmin = res[np.isfinite(res)].max(), res[np.isfinite(res)].min()
    print(vmax, vmin)
    print(ylim_max, ylim_min)


    norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)
    x, y = np.meshgrid(Xrange, line)
    print(x.shape, y.shape, res.shape)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(15, 8))
    cs = ax.plot_surface(x, y, res.T, cmap=cm.viridis, norm=norm)
    fig.colorbar(cs, ax=ax)
    ax.set_zlabel('PDF')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

    fig, ax = plt.subplots(figsize=(15, 8))
    cs = ax.contourf(x, y, res.T, levels=150)
    fig.colorbar(cs, ax=ax)
    # ax.contour(cs, colors='k')
    ax.scatter(Xrange, Y, color="gray", alpha=0.5)
    Ymean, Yvar = model.predict_y(X)
    ax.plot(Xrange, Ymean.numpy(), color="blue", lw=3, marker="x")
    plt.show()


# Main execution
# N = 2001
pred_distribution = tfp.distributions.Gamma
# X, Y = initialize_data(N)
train_data, test_data, column_names = streamflow_dataset(input_width=1)
x_test, y_test = test_data
x_train, y_train = train_data
task = 2
X, Y = x_train[6000:, task][:, None], y_train[6000:, task][:, None]
print(X[690], X[691])
model = build_model(X, Y)
# train_model(model, (X, Y), epochs=100)


def mean_squared_error(y, y_pred):
        return np.mean((y - y_pred) ** 2)

y_pred = model.predict_y(x_test[:, task][:, None])
print(mean_squared_error(y_test[:, task][:, None], y_pred))
X, Y = x_test[:, task][:, None], y_test[:, task][:, None]
plt.scatter(X, Y)
plt.show()
plot_results(model, X, Y)