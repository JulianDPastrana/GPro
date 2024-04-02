from gpflow.likelihoods import Likelihood
from tensorflow.python.framework.ops import Tensor
import tensorflow as tf
from gpflow.quadrature import NDiagGHQuadrature
import math as m
import numpy as np

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

        # Gaussian quadrature for numerical integration with TensorFlow and tnp compatibility
        
        gc = NDiagGHQuadrature(dim=1, n_gh=500)
        # Loop through each observation dimension
        for d in range(self.observation_dim):
            idx_mu_f_d_1 = d * 2
            idx_mu_f_d_2 = idx_mu_f_d_1 + 1
            
            mu_f_d_1 = Fmu[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            var_f_d_1 = Fvar[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            mu_f_d_2 = Fmu[..., idx_mu_f_d_2:idx_mu_f_d_2+1]
            var_f_d_2 = Fvar[..., idx_mu_f_d_2:idx_mu_f_d_2+1]

            # gc_mean = NDiagGHQuadrature(dim=1, n_gh=150)
            # gc_var = NDiagGHQuadrature(dim=1, n_gh=150)
            n_samples = 1000
            print(mu_f_d_2.shape)
            f_d_2_samples = np.random.normal(loc=mu_f_d_2, scale=var_f_d_2, size=n_samples)

            # Compute predicted mean for the current observation dimension using Gaussian quadrature
            pred_mean = tf.math.exp(mu_f_d_1 + 0.5 * var_f_d_1) * tf.reduce_mean(tf.math.exp(0.5 * tf.math.exp(f_d_2_samples)))#gc(lambda f: tf.math.exp(0.5 * tf.math.exp(f)), mu_f_d_2, var_f_d_2)
            pred_sqmean = tf.math.exp(2*(mu_f_d_1 + var_f_d_1)) * tf.reduce_mean(tf.math.exp(2 * tf.math.exp(f_d_2_samples)))#gc(lambda f: tf.math.exp(2 * tf.math.exp(f)), mu_f_d_2, var_f_d_2)

            predicted_means.append(pred_mean)
            predicted_variances.append(pred_sqmean - tf.math.square(pred_mean))
        
        predicted_means = tf.concat(predicted_means, axis=-1)
        predicted_variances = tf.concat(predicted_variances, axis=-1)
        # print(np.sum(np.isnan(X)))
        return predicted_means, predicted_variances
    
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError

