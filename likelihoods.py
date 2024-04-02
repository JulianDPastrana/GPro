from gpflow.likelihoods import Likelihood
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature
import math as m
import numpy as np
from scipy.stats import norm
from scipy.special import roots_hermite
import tensorflow_probability as tfp
from scipy.special import roots_legendre

class LogNormalLikelihood(Likelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim, latent_dim, observation_dim)
        self.eps = tf.cast(1e-4, tf.float64)

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
            term = tf.math.log(2*tf.cast(m.pi, tf.float64)) + 2*logYd + f_d_2 + tf.math.square(logYd - f_d_1) * tf.math.exp(-f_d_2)
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
            term = tf.math.log(2 * tf.cast(m.pi, tf.float64)) + 2 * logYd + mu_f_d_2 + tf.math.exp(-mu_f_d_2 + sigma_f_d_2 / 2) * (
                tf.math.square(logYd - mu_f_d_1) + sigma_f_d_1)
            
            # Accumulate the terms from all observation dimensions
            total_terms += term

        # Final expected log density, summing over all dimensions and data points
        expected_log_density = -0.5 * tf.reduce_sum(total_terms, axis=-1)

        return expected_log_density
        


    def _predict_mean_and_var(self, X, Fmu, Fvar):
        predicted_means = []
        predicted_variances = []

        # Loop through each observation dimension
        for d in range(self.observation_dim):
            idx_mu_f_d_1 = d * 2
            idx_mu_f_d_2 = idx_mu_f_d_1 + 1

            mu_f_d_1 = Fmu[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            var_f_d_1 = Fvar[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            mu_f_d_2 = Fmu[..., idx_mu_f_d_2:idx_mu_f_d_2+1]
            var_f_d_2 = Fvar[..., idx_mu_f_d_2:idx_mu_f_d_2+1]

            # Number of Monte Carlo samples
            n_samples = 20000  

            # Sample from q(f_{d,2})
            f_d_2_samples = np.random.normal(size=(mu_f_d_2.shape[0], n_samples), loc=mu_f_d_2, scale=tf.sqrt(var_f_d_2))

            # Compute the integral for expected value using Monte Carlo integration
            pred_mean_integral = tf.exp(mu_f_d_1 + 0.5 * var_f_d_1) * tf.reduce_mean(tf.exp(0.5 * tf.exp(f_d_2_samples)), axis=0)

            # Compute the integral for expected value of square using Monte Carlo integration
            pred_var_integral = tf.exp(2*(mu_f_d_1 + var_f_d_1)) * tf.reduce_mean(tf.exp(2 * tf.exp(f_d_2_samples)), axis=0)

            predicted_means.append(pred_mean_integral)
            predicted_variances.append(pred_var_integral - tf.math.square(pred_mean_integral))

        predicted_means = tf.concat(predicted_means, axis=-1)
        predicted_variances = tf.concat(predicted_variances, axis=-1)
        return predicted_means, predicted_variances
        
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError

