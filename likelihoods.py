from gpflow.likelihoods import Likelihood, MonteCarloLikelihood, QuadratureLikelihood
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature
import math
import numpy as np
from scipy.stats import norm
from scipy.special import roots_hermite
import tensorflow_probability as tfp
from scipy.special import roots_legendre
import gpflow as gpf

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
        f_mu = F[..., ::2]  # Extract even indices - mean
        f_sigma = F[..., 1::2]  # Extract odd indices - std deviation

        log_y = tf.math.log(tf.maximum(Y, self.eps))
        term1 = tf.reduce_sum(-0.5 * ((log_y - f_mu) ** 2) / tf.exp(f_sigma), axis=-1)
        term2 = tf.reduce_sum(-0.5 * tf.math.log(2. * math.pi * tf.exp(f_sigma)), axis=-1)
        log_prob = term1 + term2

        return log_prob

    def _conditional_mean(self, X, F) -> tf.Tensor:
        """
        Computes the conditional mean E[Y|F] for a log-normal distribution.
        """
        f_mu = F[..., ::2]  # Mean parameters
        f_sigma = F[..., 1::2]  # Variance parameters
        
        # For log-normal, E[Y|F] = exp(mu + sigma^2 / 2)
        conditional_mean = tf.exp(f_mu + 0.5 * tf.exp(f_sigma))
        return conditional_mean

    def _conditional_variance(self, X, F) -> tf.Tensor:
        """
        Computes the conditional variance Var[Y|F] for a log-normal distribution.
        """
        f_mu = F[..., ::2]
        f_sigma = F[..., 1::2]
        
        # For log-normal, Var[Y|F] = (exp(sigma^2) - 1) * exp(2 * mu + sigma^2)
        conditional_variance = (tf.exp(tf.exp(f_sigma)) - 1) * tf.exp(2 * f_mu + tf.exp(f_sigma))
        return conditional_variance

class LogNormalLikelihood(Likelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim, latent_dim, observation_dim)
        self.eps = tf.cast(1e-5, tf.float64)
        self.quadrature = NDiagGHQuadrature(dim=1, n_gh=50)

    def _log_prob(self, X, F, Y) -> Tensor:
        total_terms = 0
        # Loop through each observation dimension d
        for d in range(self.observation_dim):
            # Indices for mu and sigma of f_{d,1} and f_{d,2}
            idx_f_d_1 = d * 2
            idx_f_d_2 = idx_f_d_1 + 1
            
            f_d_1 = F[..., idx_f_d_1:idx_f_d_1+1]
            f_d_2 = F[..., idx_f_d_2:idx_f_d_2+1]

            Y_d = Y[..., d:d+1]
            logYd = tf.math.log(Y_d + self.eps)
            term = tf.math.log(2*tf.cast(math.pi, tf.float64)) + 2*logYd + f_d_2 + tf.math.square(logYd - f_d_1) * tf.math.exp(-f_d_2)
            total_terms += term
        
        
        log_probability_density = -0.5 * tf.reduce_sum(total_terms, axis=-1)

        return log_probability_density
    
    def _variational_expectations(self, X, Fmu, Fvar, Y) -> Tensor:
        total_terms = 0
        # Loop through each observation dimension d
        for d in range(self.observation_dim):
            # Indices for mu and sigma of f_{d,1} and f_{d,2}
            idx_f_d_1 = d * 2
            idx_f_d_2 = idx_f_d_1 + 1
            
            # Extracting the relevant slices for f_{d,1} (mean and variance)
            mu_f_d_1 = Fmu[..., idx_f_d_1:idx_f_d_1+1]
            sigma_f_d_1 = Fvar[..., idx_f_d_1:idx_f_d_1+1]
            
            # Extracting the relevant slices for f_{d,2} (mean and variance)
            mu_f_d_2 = Fmu[..., idx_f_d_2:idx_f_d_2+1]
            sigma_f_d_2 = Fvar[..., idx_f_d_2:idx_f_d_2+1]
            
            Y_d = Y[..., d:d+1]
            # Compute the terms according to the expected log density formula for each observation dimension d
            logYd = tf.math.log(Y_d + self.eps)
            term = tf.math.log(2 * tf.cast(math.pi, tf.float64)) + 2 * logYd + mu_f_d_2 + tf.math.exp(-mu_f_d_2 + sigma_f_d_2 / 2) * (
                tf.math.square(logYd - mu_f_d_1) + sigma_f_d_1)
            
            # Accumulate the terms from all observation dimensions
            total_terms += term

        # Final expected log density, summing over all dimensions and data points
        expected_log_density = -0.5 * tf.reduce_sum(total_terms, axis=-1)

        return expected_log_density
        

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        def integrand_mean(f):
            return tf.exp(0.5*tf.exp(f))

        def integrand_variance(f):
            return tf.exp(2*tf.exp(f))

        predicted_means = []
        predicted_variances = []

        for d in range(self.observation_dim):
            idx_mu_f_d_1 = d * 2
            idx_mu_f_d_2 = idx_mu_f_d_1 + 1

            mu_f_d_1 = Fmu[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            var_f_d_1 = Fvar[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            mu_f_d_2 = Fmu[..., idx_mu_f_d_2:idx_mu_f_d_2+1]
            var_f_d_2 = Fvar[..., idx_mu_f_d_2:idx_mu_f_d_2+1]

            pred_mean = tf.exp(mu_f_d_1 + 0.5 * var_f_d_1) * self.quadrature(integrand_mean, mu_f_d_2, var_f_d_2)
            pred_var = tf.exp(2 * mu_f_d_1 + 2 * var_f_d_1) * self.quadrature(integrand_variance, mu_f_d_2, var_f_d_2)

            predicted_means.append(pred_mean)
            predicted_variances.append(pred_var - tf.square(pred_mean))

        predicted_means = tf.concat(predicted_means, axis=-1)
        predicted_variances = tf.concat(predicted_variances, axis=-1)

        return predicted_means, predicted_variances
        
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError

