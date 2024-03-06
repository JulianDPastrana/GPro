from gpflow.base import MeanAndVariance
from gpflow.likelihoods import Likelihood
from tensorflow.python.framework.ops import Tensor
from numpy import ndarray
import tensorflow as tf
import numpy as np
from gpflow.quadrature import GaussianQuadrature, NDiagGHQuadrature
from gpflow.quadrature.gauss_hermite import gh_points_and_weights


class LogNormalLikelihood(Likelihood):
    def __init__(self, input_dim, latent_dim, observation_dim) -> None:
        super().__init__(input_dim=1, latent_dim=2, observation_dim=1)

    def _log_prob(self, X, F, Y) -> Tensor:
        eps = 1e-6
        mu = F[..., :1]
        sigma = tf.math.exp(F[..., 1:])
        norm_term = -tf.math.log(Y+eps)-tf.math.log(2*np.pi)-0.5*tf.math.log(sigma)
        unorm_term = -0.5 * tf.square(tf.math.log(Y+eps) - mu) / sigma
        return tf.reduce_sum(norm_term + unorm_term, axis=-1)
    
    def _variational_expectations(self, X, Fmu, Fvar, Y) -> Tensor:
        eps = np.float64(1e-6)
        muf = Fmu[..., :1]
        mug = Fmu[..., 1:]
        sigmaf = Fvar[..., :1]
        sigmag = Fvar[..., 1:]
        A = -tf.math.log(Y+eps)-tf.math.log(2*np.float64(np.pi))
        B = -0.5*mug - 0.5*tf.math.log(Y+eps)**2 * tf.math.exp(-mug + sigmag/2) + tf.math.log(Y+eps)*muf*tf.math.exp(-mug + sigmag/2)
        C = -0.5 * (sigmaf**2 + muf**2) * tf.math.exp(-mug + sigmag/2)
        return tf.reduce_sum(A+B+C, axis=-1)
    
    def _predict_mean_and_var(self, X, Fmu, Fvar):
        mean_f = Fmu[..., :1]
        mean_g = Fmu[..., 1:]
        var_f = Fvar[..., :1]
        var_g = Fvar[..., 1:]
        gc = NDiagGHQuadrature(dim=1, n_gh=50)

        func_mean = lambda g: tf.math.exp(0.5 * tf.math.exp(g))
        pred_mean = tf.math.exp(mean_f + 0.5 * var_f)*gc(func_mean, mean_g, var_g)

        func_var = lambda g: tf.math.exp(2 * tf.math.exp(g))
        pred_var = tf.math.exp(2*(mean_f+var_f))*gc(func_var, mean_g, var_g) - pred_mean**2
        return pred_mean, pred_var
    
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError


likelihood = LogNormalLikelihood(input_dim=1, latent_dim=2, observation_dim=1)