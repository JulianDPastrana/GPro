#!/home/usuario/Documents/Gpro/gpvenv/bin/python3
from typing import Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.likelihoods import MultiLatentTFPConditional
from tqdm import trange
from gpflow.utilities import tabulate_module_summary
from check_shapes import check_shapes, inherit_check_shapes
from miscellaneous import *
from datasets import *


class HeteroskedasticLikelihood(MultiLatentTFPConditional):
    
    def __init__(
        self,
        distribution_class,
        param1_transform,
        param2_transform,
        **kwargs: Any,
    ) -> None:
        
        self.param1_transform = param1_transform
        self.param2_transform = param2_transform
        @check_shapes(
            "F: [batch..., 2]",
        )
        def conditional_distribution(F) -> tfp.distributions.Distribution:
            param1 = self.param1_transform(F[..., :1])
            param2 = self.param2_transform(F[..., 1:])
            return distribution_class(4, param1, param2)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )

# DATA COLLECTION

train_data, test_data = streamflow_dataset()
x_train, y_train = train_data
x_test, y_test = test_data

# PLOT DATA

def plot_distribution(X, Y, Ymean, Yvar, title_for_save=None):
    plt.figure(figsize=(15, 5))
    # plt.plot(X, Ymean)
    # plt.plot(X, Yvar)
    plt.scatter(X, Y, color="red", alpha=0.8)
    loc = Ymean
    scale = np.sqrt(Yvar)
    # x = X.squeeze()
    for k in (1, 2):
        lb = (loc - k * scale).squeeze()
        ub = (loc + k * scale).squeeze()
        plt.fill_between(X, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3)
    plt.plot(X, lb, color="silver")
    plt.plot(X, ub, color="silver")
    plt.plot(X, loc, color="black")
    # plt.xlim(6000, 7000)

    # plt.legend()

    if title_for_save:
        plt.title(title_for_save)
        plt.savefig(f"./train_step/{title_for_save}.png")
    plt.close()


# BUILD MODEL

# Likelihood

likelihood = HeteroskedasticLikelihood(
    distribution_class=tfp.distributions.StudentT,
    param1_transform=tfp.bijectors.Identity(),
    param2_transform=tfp.bijectors.Exp()
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
        Yvar = Yvar.numpy().squeeze()
        Ystd = np.sqrt(Yvar)
        # Psamples = tfp.distributions.LogNormal(Ymean, Ystd).sample(10).numpy().T
        plot_distribution(range(len(y_train)), y_train, Ymean, Yvar,
                          f"Epoch {epoch} - " + loss_text)


# SAVE THINGS

with open("parameters_summary.txt", "w") as file:
    file.write(tabulate_module_summary(model))

create_video("training_summary.avi", "./train_step")
