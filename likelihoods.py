from gpflow.likelihoods import Likelihood, MonteCarloLikelihood, QuadratureLikelihood
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature, ndiag_mc, ndiagquad
import math
import numpy as np
from scipy.stats import norm
from scipy.special import roots_hermite
import tensorflow_probability as tfp
from scipy.special import roots_legendre
import gpflow as gpf
from check_shapes import check_shapes
from gpflow.likelihoods import MultiLatentTFPConditional

class LogNormalQuadLikelihood(QuadratureLikelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(
            input_dim=input_dim,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            quadrature=NDiagGHQuadrature(latent_dim, 10)
        )
        self.eps = tf.cast(1e-6, dtype=gpf.default_float())

    def _log_prob(self, X, F, Y) -> tf.Tensor:
        """
        Calculates the log probability of Y given F for a log-normal distribution.
        """
        # Split F into mean and log variance components for each output dimension
        f_mu = F[..., ::2]  # Even indices: mean
        f_sigma = F[..., 1::2]  # Odd indices: log variance

        # Ensure Y is positive to avoid log of non-positive numbers
        Y = tf.maximum(Y, self.eps)
        logY = tf.math.log(Y)

        # Log-normal log probability density function
        term1 = tf.reduce_sum(-0.5 * ((logY - f_mu) ** 2) / tf.exp(f_sigma), axis=-1)
        term2 = tf.reduce_sum(-0.5 * tf.math.log(2. * math.pi * tf.exp(f_sigma)), axis=-1)
        log_prob = term1 + term2
        return log_prob

    def _conditional_mean(self, X, F) -> tf.Tensor:
        """
        Computes the conditional mean E[Y|F] of the log-normal distribution.
        """
        f_mu = F[..., ::2]  # Mean of the log-normal distribution
        return tf.exp(f_mu + 0.5 * F[..., 1::2])  # E[Y|F] = exp(mu + sigma^2 / 2)

    def _conditional_variance(self, X, F) -> tf.Tensor:
        """
        Computes the conditional variance Var[Y|F] of the log-normal distribution.
        """
        f_mu = F[..., ::2]
        f_sigma = F[..., 1::2]
        var = (tf.exp(f_sigma) - 1) * tf.exp(2 * f_mu + f_sigma)  # Var[Y|F] formula for log-normal
        return var

class LogNormalMCLikelihood(MonteCarloLikelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim, latent_dim, observation_dim)
        self.num_monte_carlo_points = 500
        self.eps = tf.cast(1e-6, tf.float64)

    def _log_prob(self, X, F, Y) -> tf.Tensor:
        """
        Computes the log probability density log p(Y|F) for a log-normal distribution.
        """
        Fd1 = F[..., ::2]  # Extract even indices - mean
        Fd2 = F[..., 1::2]  # Extract odd indices - std deviation
        log_y = tf.math.log(tf.maximum(Y, self.eps))
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * tf.cast(math.pi, tf.float64)) + 2 * log_y + tf.math.log(tf.math.softplus(Fd2)) + (log_y - Fd1) ** 2 / tf.math.softplus(Fd2),
            axis=-1
        )

    def _conditional_mean(self, X, F) -> tf.Tensor:
        """
        Computes the conditional mean E[Y|F] for a log-normal distribution.
        """
        f_mu = F[..., ::2]  # Mean parameters
        f_sigma = F[..., 1::2]  # Variance parameters
        
        # For log-normal, E[Y|F] = exp(mu + sigma^2 / 2)
        conditional_mean = tf.exp(f_mu + 0.5 * tf.math.softplus(f_sigma))
        return conditional_mean

    def _conditional_variance(self, X, F) -> tf.Tensor:
        """
        Computes the conditional variance Var[Y|F] for a log-normal distribution.
        """
        f_mu = F[..., ::2]
        f_sigma = F[..., 1::2]
        
        # For log-normal, Var[Y|F] = (exp(sigma^2) - 1) * exp(2 * mu + sigma^2)
        conditional_variance = (tf.exp(tf.math.softplus(f_sigma)) - 1) * tf.exp(2 * f_mu + tf.math.softplus(f_sigma))
        return conditional_variance

class LogNormalLikelihood(Likelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim, latent_dim, observation_dim)
        self.eps = tf.cast(1e-6, tf.float64)

    def _log_prob(self, X, F, Y) -> Tensor:
        """
        Computes the log probability density log p(Y|F) for a log-normal distribution.
        """
        Fd1 = F[..., ::2]  # Extract even indices - mean
        Fd2 = F[..., 1::2]  # Extract odd indices - std deviation
        log_y = tf.math.log(tf.maximum(Y, self.eps))
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * tf.cast(math.pi, tf.float64)) + 2 * log_y + Fd2 + (log_y - Fd1) ** 2 / tf.exp(Fd2),
            axis=-1
        )
    
    def _variational_expectations(self, X, Fmu, Fvar, Y) -> Tensor:
        Fd1mu = Fmu[..., ::2] 
        Fd2mu = Fmu[..., 1::2]
        Fd1var = Fvar[..., ::2] 
        Fd2var = Fvar[..., 1::2] 
        log_y = tf.math.log(tf.maximum(Y, self.eps))
        return -0.5 * tf.reduce_sum(
            tf.math.log(2 * tf.cast(math.pi, tf.float64)) + 2 * log_y + Fd2mu + tf.exp(-Fd2mu + Fd2var / 2) * ((log_y - Fd1mu) ** 2 + Fd1var),
            axis=-1
        )
        

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        Fd1mu = Fmu[..., ::2] 
        Fd2mu = Fmu[..., 1::2]
        Fd1var = Fvar[..., ::2] + self.eps
        Fd2var = Fvar[..., 1::2] + self.eps

        def integrand_mean(f):
            return tf.exp(0.5 * tf.exp(f))

        def integrand_variance(f):
            return tf.exp(2 * tf.exp(f))

        # Vectorized computation of predicted means and variances
        pred_mean = tf.exp(Fd1mu + 0.5 * Fd1var) * ndiagquad(integrand_mean, 150, Fd2mu, Fd2var)
        pred_var = tf.exp(2 * Fd1mu + 2 * Fd1var) * ndiagquad(integrand_variance, 150, Fd2mu, Fd2var) - tf.square(pred_mean)

        print("NaNs", np.isnan(pred_mean.numpy()).sum())
        # Since `pred_mean` and `pred_var` are already concatenated across the observation dimension, no need for further concatenation
        return pred_mean, pred_var

    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError



class HeteroskedasticLikelihood(MultiLatentTFPConditional):

    def __init__(
        self,
        distribution_class,
        param1_transform,
        param2_transform,
        # **kwargs: Any,
    ) -> None:

        self.param1_transform = param1_transform
        self.param2_transform = param2_transform
        self.distribution_class = distribution_class
        @check_shapes(
            "F: [batch..., 2]",
        )
        def conditional_distribution(F) -> tfp.distributions.Distribution:
            param1 = self.param1_transform(F[..., :1])
            param2 = self.param2_transform(F[..., 1:])
            return self.distribution_class(param1, param2)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            # **kwargs,
        )