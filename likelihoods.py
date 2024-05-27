from gpflow.likelihoods import MonteCarloLikelihood
from typing import Optional, Type
import tensorflow_probability as tfp
import tensorflow as tf


class MOChainedLikelihoodMC(MonteCarloLikelihood):
    """
    A Monte Carlo Likelihood class for multi-output models using a chained likelihood approach.
    This class models the likelihood of observations given latent functions using Monte Carlo approximation.

    Attributes:
        param1_transform (Optional[tfp.bijectors.Bijector]): A bijector transforming the first parameter, typically used to ensure the parameter is in the correct domain.
        param2_transform (Optional[tfp.bijectors.Bijector]): A bijector transforming the second parameter, ensuring it is in the correct domain.
        distribution_class (Type[tfp.distributions.Distribution]): The distribution class used for modeling the likelihood.
        num_monte_carlo_points (int): The number of Monte Carlo points used for approximation.
    """

    def __init__(self, 
                 input_dim: int, 
                 latent_dim: int, 
                 observation_dim: int,
                 distribution_class: Type[tfp.distributions.Distribution],
                 param1_transform: Optional[tfp.bijectors.Bijector] = None,
                 param2_transform: Optional[tfp.bijectors.Bijector] = None) -> None:
        """
        Initializes the MOChainedLikelihoodMC class.

        Args:
            input_dim (int): The dimension of the input space.
            latent_dim (int): The dimension of the latent space.
            observation_dim (int): The dimension of the observation space.
            distribution_class (type): The distribution class to be used.
            param1_transform (Callable): Transformation function for the first parameter.
            param2_transform (Callable): Transformation function for the second parameter.
        """
        self.param1_transform = param1_transform
        self.param2_transform = param2_transform
        self.distribution_class = distribution_class
        self.num_monte_carlo_points = 1500
        super().__init__(input_dim, latent_dim, observation_dim)

    def _log_prob(self, X: tf.Tensor, F: tf.Tensor, Y: tf.Tensor) -> tf.Tensor:
        """
        Computes the log probability density log p(Y|F).

        Args:
            X (tf.Tensor): Input data tensor.
            F (tf.Tensor): Latent function values tensor.
            Y (tf.Tensor): Observed data tensor.

        Returns:
            tf.Tensor: Log probability density tensor.
        """
        Fd1 = F[..., ::2]  # Extract even indices - mean
        Fd2 = F[..., 1::2]  # Extract odd indices - standard deviation
        alpha = self.param1_transform(Fd1)
        beta = self.param2_transform(Fd2)
        dist = self.distribution_class(alpha, beta, force_probs_to_zero_outside_support=True)
        return tf.reduce_sum(dist.log_prob(Y), axis=-1)

    def _conditional_mean(self, X: tf.Tensor, F: tf.Tensor) -> tf.Tensor:
        """
        Computes the conditional mean E[Y|F].

        Args:
            X (tf.Tensor): Input data tensor.
            F (tf.Tensor): Latent function values tensor.

        Returns:
            tf.Tensor: Conditional mean tensor.
        """
        Fd1 = F[..., ::2]  # Extract even indices - mean
        Fd2 = F[..., 1::2]  # Extract odd indices - variance
        alpha = self.param1_transform(Fd1)
        beta = self.param2_transform(Fd2)
        dist = self.distribution_class(alpha, beta, force_probs_to_zero_outside_support=True)
        return dist.mean()

    def _conditional_variance(self, X: tf.Tensor, F: tf.Tensor) -> tf.Tensor:
        """
        Computes the conditional variance Var[Y|F].

        Args:
            X (tf.Tensor): Input data tensor.
            F (tf.Tensor): Latent function values tensor.

        Returns:
            tf.Tensor: Conditional variance tensor.
        """
        Fd1 = F[..., ::2]  # Extract even indices - mean
        Fd2 = F[..., 1::2]  # Extract odd indices - variance
        alpha = self.param1_transform(Fd1)
        beta = self.param2_transform(Fd2)
        dist = self.distribution_class(alpha, beta, force_probs_to_zero_outside_support=True)
        return dist.variance()
