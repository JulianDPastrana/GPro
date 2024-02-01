import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.likelihoods import MultiLatentTFPConditional
from check_shapes import check_shapes
from typing import Any, Callable
import matplotlib

import gpflow as gpf

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
            return distribution_class(param1, param2)

        super().__init__(
            latent_dim=2,
            conditional_distribution=conditional_distribution,
            **kwargs,
        )

N = 1001

np.random.seed(0)
tf.random.set_seed(0)

# Build inputs X
X = np.linspace(0, 4 * np.pi, N)[:, None]  # X must be of shape [N, 1]

# Deterministic functions in place of latent ones
f1 = np.sin
f2 = np.cos

# Use transform = exp to ensure positive-only scale values
transform = np.exp

# Compute loc and scale as functions of input X
alpha = transform(f1(X))
beta = transform(f2(X))

# Sample outputs Y from Gaussian Likelihood
Y = np.random.beta(alpha, beta)




likelihood = HeteroskedasticLikelihood(
    distribution_class=tfp.distributions.Beta,  # Gaussian Likelihood
    param1_transform=tfp.bijectors.Exp(),  # Exponential Transform
    param2_transform=tfp.bijectors.Exp(),  # Exponential Transform
)

print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

kernel = gpf.kernels.SeparateIndependent(
    [
        gpf.kernels.SquaredExponential(),  # This is k1, the kernel of f1
        gpf.kernels.SquaredExponential(),  # this is k2, the kernel of f2
    ]
)
# The number of kernels contained in gpf.kernels.SeparateIndependent must be the same as likelihood.latent_dim
M = 20  # Number of inducing variables for each f_i

# Initial inducing points position Z
Z = np.linspace(X.min(), X.max(), M)[:, None]  # Z must be of shape [M, 1]

inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
    [
        gpf.inducing_variables.InducingPoints(Z),  # This is U1 = f1(Z1)
        gpf.inducing_variables.InducingPoints(Z),  # This is U2 = f2(Z2)
    ]
)

model = gpf.models.SVGP(
    kernel=kernel,
    likelihood=likelihood,
    inducing_variable=inducing_variable,
    num_latent_gps=likelihood.latent_dim,
)

data = (X, Y)
loss_fn = model.training_loss_closure(data)

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

epochs = 100
log_freq = 20

for epoch in range(1, epochs + 1):
    optimisation_step()

    # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    if epoch % log_freq == 0 and epoch > 0:
        print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")
        # Ymean, Yvar = model.predict_y(X)
        # Ymean = Ymean.numpy().squeeze()
        # Yvar = Yvar.numpy().squeeze()
        # plot_distribution(X, Y, model)


num_samples = 60

num_y_pixels = 60
#Want the top left pixel to be evaluated at 1
from scipy.stats import beta as beta_dist
Fsamples = model.predict_f_samples(X, num_samples)
Ysamples = model.likelihood.conditional_distribution(Fsamples)
alpha = np.exp(Fsamples[..., 0].numpy().squeeze())
beta = np.exp(Fsamples[..., 1].numpy().squeeze())
line = np.linspace(1, 0, num_y_pixels)
res = np.zeros((X.shape[0], num_y_pixels))
for j in range(X.shape[0]):
    sf = alpha[:, j]  # Pick out the jth point along X axis
    sg = beta[:, j]
    for i in range(num_samples):
        # Pick out the sample and evaluate the pdf on a line between 0
        # and 1 with these alpha and beta values
        res[j, :] += beta_dist.pdf(line, sf[i], sg[i])
    res[j, :] /= num_samples

vmax, vmin = res[np.isfinite(res)].max(), res[np.isfinite(res)].min()


norm = matplotlib.colors.Normalize(vmax=vmax, vmin=vmin)


fig3d = plt.figure(figsize=(13,5))
ax = fig3d.add_subplot(111, projection='3d')
# ax.view_init(elev=55., azim=300.0)
axlim_min, axlim_max = X[:, 0].min(), X[:, 0].max()
x, y = np.mgrid[axlim_min:axlim_max:complex(res.shape[0]),
                1:0:complex(res.shape[1])]
#x_dates = num2date(x)
# xfmt = mdates.DateFormatter('%b %d')
ax.plot_surface(x,y,res,rstride=1, cmap=plt.cm.YlOrRd, cstride=1, lw=0.05, alpha=1, edgecolor='b', norm=norm)
# #ax.xaxis.set_major_formatter(xfmt)
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d/%Y'))
# ax.xaxis.set_major_locator(mdates.DayLocator())
ax.set_zlabel('PDF')
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

plt.contourf(x, y, res, levels=10)
plt.colorbar()
plt.scatter(X, Y, color="gray", alpha=0.8)
Ymean, Yvar = model.predict_y(X)
plt.plot(X, Ymean.numpy(), color="blue", lw=5)
plt.show()