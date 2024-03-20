from gpflow.base import MeanAndVariance
from gpflow.likelihoods import Likelihood
from tensorflow.python.framework.ops import Tensor
from numpy import ndarray
import tensorflow as tf
import numpy as np
from gpflow.quadrature import GaussianQuadrature, NDiagGHQuadrature
from gpflow.quadrature.gauss_hermite import gh_points_and_weights
import tensorflow.experimental.numpy as tnp


class LogNormalLikelihood(Likelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim, latent_dim, observation_dim)
        # self.input_dim = input_dim
        # self.latent_dim = latent_dim
        # self.observation_dim = observation_dim
        self.eps = tf.cast(1e-6, tf.float64)

    def _log_prob(self, X, F, Y) -> Tensor:
        # mu = F[..., :1]
        # sigma = tnp.exp(F[..., 1:])
        # norm_term = -tnp.log(Y+self.eps)-0.5*tnp.log(2*np.pi)-0.5*tnp.log(sigma)
        # unorm_term = -0.5 * tnp.square(tnp.log(Y+self.eps) - mu) / sigma
        # return tf.reduce_sum(norm_term + unorm_term, axis=-1)
        raise NotImplementedError
    
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
            
            # Compute the terms according to the expected log density formula for each observation dimension d
            term = tnp.log(2 * tnp.pi) + 2 * tnp.log(Y[..., d:d+1] + self.eps) + mu_f_d_2 + tnp.exp(-mu_f_d_2 + sigma_f_d_2 / 2) * (
                tnp.square(tf.math.log(Y[..., d:d+1] + self.eps)) - 2 * tnp.log(Y[..., d:d+1] + self.eps) * mu_f_d_1 + tnp.square(mu_f_d_1) + sigma_f_d_1)
            
            # Accumulate the terms from all observation dimensions
            total_terms += term
        
        # Final expected log density, summing over all dimensions and data points
        expected_log_density = -0.5 * tf.reduce_sum(total_terms, axis=-1)
        
        return expected_log_density
    
    def _predict_mean_and_var(self, X, Fmu, Fvar):
        predicted_means = []

        # Gaussian quadrature for numerical integration with TensorFlow and tnp compatibility
        

        # Loop through each observation dimension
        for d in range(self.observation_dim):
            idx_mu_f_d_1 = d * 2
            idx_mu_f_d_2 = idx_mu_f_d_1 + 1
            
            mu_f_d_1 = Fmu[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            var_f_d_1 = Fvar[..., idx_mu_f_d_1:idx_mu_f_d_1+1]
            mu_f_d_2 = Fmu[..., idx_mu_f_d_2:idx_mu_f_d_2+1]
            var_f_d_2 = Fvar[..., idx_mu_f_d_2:idx_mu_f_d_2+1]

            gc = NDiagGHQuadrature(dim=1, n_gh=50)
            # Function to be integrated over f_{d,2}
            func_mean = lambda f_d_2: tnp.exp(0.5 * tnp.exp(f_d_2))

            # Compute predicted mean for the current observation dimension using Gaussian quadrature
            pred_mean = tnp.exp(mu_f_d_1 + 0.5 * var_f_d_1) * gc(func_mean, mu_f_d_2, var_f_d_2)

            predicted_means.append(pred_mean)
        
        predicted_means = tf.concat(predicted_means, axis=-1)
        predicted_variances = None  # Placeholder for variance computation

        return predicted_means, predicted_variances
    
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError


likelihood = LogNormalLikelihood(input_dim=1, latent_dim=2, observation_dim=1)