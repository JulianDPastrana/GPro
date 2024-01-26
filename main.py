#!/home/usuario/Documents/Gpro/gpvenv/bin/python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import trange
from gpflow.utilities import tabulate_module_summary
from miscellaneous import *
from datasets import *

# DATA COLLECTION

train_data, test_data = lognorm_datset(
    input_width=1,
    label_width=1,
    shift=1,
    N=1001
)
x_train, y_train = train_data
x_test, y_test = test_data

# PLOT DATA

def plot_distribution(X, Y, loc, scale, title_for_save=None):
    plt.figure(figsize=(15, 5))
    x = X.squeeze()
    for k in range(1, 4):
        # lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        plt.fill_between(x, ub, color="C0", alpha=0.3, label=f"$\pm {k}\sigma$")

    plt.plot(X, loc, color="C0", label="Mean function")
    plt.scatter(X, Y, color="gray", alpha=0.8)

    plt.legend()

    if title_for_save:
        plt.title(title_for_save)
        plt.savefig(f"./train_step/{title_for_save}.png")
    plt.close()


# BUILD MODEL

# Likelihood
likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    distribution_class=tfp.distributions.LogNormal,  # Gaussian Likelihood
    scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

kernel = gpf.kernels.LinearCoregionalization(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ],
    W=np.random.randn(likelihood.latent_dim, likelihood.latent_dim),
)

# The number of kernels contained in the kernel must be the same as likelihood.latent_dim

# Inducing Points
M = 20  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(x_train.min(), x_train.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
    [
        gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
)

# SVGP Model = Kenernel + Likelihood + Inducing Variables
model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)


# MODEL OPTIMIZATION

# Build Optimizers (NatGrad + Adam)
loss_fn = model.training_loss_closure(train_data)

gpf.utilities.set_trainable(model.q_mu, False)
gpf.utilities.set_trainable(model.q_sqrt, False)

variational_vars = [(model.q_mu, model.q_sqrt)]
natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

adam_vars = model.trainable_variables
adam_opt = tf.optimizers.Adam(0.01)


@tf.function
def optimisation_step():
    natgrad_opt.minimize(loss_fn, variational_vars)
    adam_opt.minimize(loss_fn, adam_vars)


# Run Optimization Loop
epochs = 100
log_freq = 1

empty_a_folder("./train_step")
pbar = trange(1, epochs + 1)
for epoch in pbar:
    optimisation_step()
    loss_text = f"Loss: {loss_fn().numpy() : .4f}"
    pbar.set_description(loss_text)
    # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    if epoch % log_freq == 0 and epoch > 0:
        Ymean, Yvar = model.predict_y(x_train)
        Ymean = Ymean.numpy().squeeze()
        Ystd = tf.sqrt(Yvar).numpy().squeeze()
        plot_distribution(np.arange(len(y_train)), y_train, Ymean, Ystd,
                          f"Epoch {epoch} - " + loss_text)


# SAVE THINGS

with open("parameters_summary.txt", "w") as file:
    file.write(tabulate_module_summary(model))

create_video("training_summary.avi", "./train_step")
