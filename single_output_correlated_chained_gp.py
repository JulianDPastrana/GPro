from typing import Any
import numpy as np
import gpflow as gpf
import tensorflow_probability as tfp
from gpflow.likelihoods import MultiLatentTFPConditional
from gpflow.utilities import tabulate_module_summary, positive
from check_shapes import check_shapes
from tqdm import trange
from miscellaneous import *
from datasets import *


class HeteroskedasticLikelihood(MultiLatentTFPConditional):
    """
    Custom likelihood class for heteroskedastic modeling using GPFlow.

    Args:
        distribution_class (tfp.distributions.Distribution): The distribution class to use.
        param1_transform (tfp.bijectors.Bijector): The transform for the first parameter.
        param2_transform (tfp.bijectors.Bijector): The transform for the second parameter.
        **kwargs: Additional keyword arguments.

    Attributes:
        param1_transform (tfp.bijectors.Bijector): The transform for the first parameter.
        param2_transform (tfp.bijectors.Bijector): The transform for the second parameter.
    """

    def __init__(
        self,
        distribution_class,
        param1_transform,
        param2_transform,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the HeteroskedasticLikelihood.

        Args:
            distribution_class (tfp.distributions.Distribution): The distribution class to use.
            param1_transform (tfp.bijectors.Bijector): The transform for the first parameter.
            param2_transform (tfp.bijectors.Bijector): The transform for the second parameter.
            **kwargs: Additional keyword arguments.
        """
        if param1_transform is None:
            param1_transform = positive(base="exp")
        if param2_transform is None:
            param2_transform = positive(base="exp")
        self.param1_transform = param1_transform
        self.param2_transform = param2_transform

        @check_shapes(
            "F: [batch..., 2]",
        )
        def conditional_distribution(F) -> tfp.distributions.Distribution:
            param1 = self.param1_transform(F[..., :1])
            param2 = self.param2_transform(F[..., 1:])
            return distribution_class(param1, param2, validate_args=True)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )


class SOSVGP(gpf.models.SVGP):
    """
    Custom SVGP (Sparse Variational Gaussian Process) model with a specific configuration.

    Args:
        input_dimension (int): The input dimension of the model.
        inducing_points_position (np.ndarray): The initial inducing points position.
        likelihood_class (tfp.distributions.Distribution): The likelihood class. Default is LogNormal.

    Attributes:
        likelihood (HeteroskedasticLikelihood): The custom heteroskedastic likelihood.
        kernel (gpf.kernels.LinearCoregionalization): The linear coregionalization kernel.
        inducing_variable (gpf.inducing_variables.SeparateIndependentInducingVariables): The inducing variable.
    """

    def __init__(self, input_dimension, inducing_points_position, likelihood_class=tfp.distributions.LogLogistic):
        """
        Initializes the CustomSVGP.

        Args:
            input_dimension (int): The input dimension of the model.
            inducing_points_position (np.ndarray): The initial inducing points position.
            likelihood_class (tfp.distributions.Distribution): The likelihood class. Default is LogNormal.
        """
        likelihood = HeteroskedasticLikelihood(
            distribution_class=likelihood_class,
            param1_transform=tfp.bijectors.Identity(),
            param2_transform=tfp.bijectors.Exp()
        )

        kernel = gpf.kernels.LinearCoregionalization(
            [
                gpf.kernels.SquaredExponential(lengthscales=[1]*input_dimension),  # This is k1, the kernel of f1
                gpf.kernels.SquaredExponential(lengthscales=[1]*input_dimension),  # this is k2, the kernel of f2
            ],
            W=np.random.randn(likelihood.latent_dim, likelihood.latent_dim),
        )

        inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
            [
                gpf.inducing_variables.InducingPoints(inducing_points_position),  # This is U1 = f1(Z1)
                gpf.inducing_variables.InducingPoints(inducing_points_position),  # This is U2 = f2(Z2)
            ]
        )

        super().__init__(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            num_latent_gps=likelihood.latent_dim,
        )

    def train_model(self, train_data, epochs=100, log_freq=1, callback=None):
        """
        Trains the CustomSVGP model.

        Args:
            train_data: The training data.
            epochs (int): The number of training epochs.
            log_freq (int): The frequency at which to log information.
            callback (callable): A callback function.

        Returns:
            None
        """
        if not callback:
            def callback():
                pass

        # MODEL OPTIMIZATION

        # Build Optimizers (NatGrad + Adam)
        loss_fn = self.training_loss_closure(train_data)

        gpf.utilities.set_trainable(self.q_mu, False)
        gpf.utilities.set_trainable(self.q_sqrt, False)

        variational_vars = [(self.q_mu, self.q_sqrt)]
        natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)

        adam_vars = self.trainable_variables
        adam_opt = tf.optimizers.Adam(0.01)

        @tf.function
        def optimisation_step():
            natgrad_opt.minimize(loss_fn, variational_vars)
            adam_opt.minimize(loss_fn, adam_vars)

        # Run Optimization Loop
        pbar = trange(1, epochs + 1)
        for epoch in pbar:
            optimisation_step()
            loss_text = f"Loss: {loss_fn().numpy() : .4f}"
            pbar.set_description(loss_text)
            # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
            if epoch % log_freq == 0 and epoch > 0:
                callback()


def main():
    train_data, _ = toy_datset()
    x_train, y_train = train_data
    n_samples, input_dimension = x_train.shape
    M = 25  # Number of inducing variables for each f_i

    # Initial inducing points position Z
    Z = np.linspace(x_train.min(), x_train.max(), M)[:, None]  # Z must be of shape [M, 1]

    # Create the CustomSVGP instance
    custom_model = SOSVGP(input_dimension=input_dimension, inducing_points_position=Z.copy())

    # Train the model
    custom_model.train_model(train_data)


if __name__ == "__main__":
    main()
