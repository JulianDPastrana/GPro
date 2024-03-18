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
        super().__init__(input_dim=1, latent_dim=2, observation_dim=1)
        self.eps = tf.cast(1e-6, tf.float64)

    def _log_prob(self, X, F, Y) -> Tensor:
        # mu = F[..., :1]
        # sigma = tnp.exp(F[..., 1:])
        # norm_term = -tnp.log(Y+self.eps)-0.5*tnp.log(2*np.pi)-0.5*tnp.log(sigma)
        # unorm_term = -0.5 * tnp.square(tnp.log(Y+self.eps) - mu) / sigma
        # return tf.reduce_sum(norm_term + unorm_term, axis=-1)
        raise NotImplementedError
    
    def _variational_expectations(self, X, Fmu, Fvar, Y) -> Tensor:
        muf = Fmu[..., :1]
        mug = Fmu[..., 1:]
        sigmaf = Fvar[..., :1]
        sigmag = Fvar[..., 1:]
        term1 = tnp.log(2*tnp.pi) + 2*tnp.log(Y+self.eps) + mug
        term2 = tnp.exp(-mug + sigmag/2) * (tnp.square(tf.math.log(Y+self.eps)) - 2*tnp.log(Y+self.eps)*muf + sigmaf + tnp.square(muf))
        return -0.5 * tf.reduce_sum(term1+term2, axis=-1)
    
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