import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood

def initialize_data(N):
    # Build inputs X
    X = np.linspace(0, 2 * np.pi, N)[:, None]  # X must be of shape [N, 1]

    # Deterministic functions in place of latent ones
    f1 = np.sin
    f2 = np.cos

    # Use transform = exp to ensure positive-only scale values
    transform = np.exp

    # Compute loc and scale as functions of input X
    loc = 0.5 * f1(X) + 2.5
    scale = transform(0.001*f2(X))

    # Sample outputs Y from LogNornal Likelihood
    Y = np.exp(np.random.normal(loc, scale))
    # Y = np.random.normal(loc, 0.2)
    # Y = np.maximum(Y, 0)
    return X, Y

def build_model(X, Y):
    """
    Builds and returns the GP model.
    """
    likelihood = LogNormalLikelihood(input_dim=1, latent_dim=2, observation_dim=1)
    kernel = gpf.kernels.SeparateIndependent(
        [gpf.kernels.SquaredExponential(), gpf.kernels.SquaredExponential()]
    )
    print("Obsevation dim: ", likelihood.observation_dim)

    M = 25
    random_indexes = np.random.choice(range(X.shape[0]), size=M, replace=False)
    Z = X[random_indexes]

    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [gpf.inducing_variables.InducingPoints(Z), gpf.inducing_variables.InducingPoints(Z)]
    )

    # # initialize mean of variational posterior to be of shape MxL
    # q_mu = np.zeros((M, 2))
    # # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    # q_sqrt = np.repeat(np.eye(M)[None, ...], 2, axis=0) * 1.0

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
        # q_mu=q_mu,
        # q_sqrt=q_sqrt
    )

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    return model

def train_model(model, data, epochs=100, log_freq=20):
    """
    Trains the model for a specified number of epochs.
    """
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

    for epoch in range(1, epochs + 1):
        optimisation_step()

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % log_freq == 0 and epoch > 0:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")


# Main execution
N = 1000
X, Y = initialize_data(N)
plt.hist(Y, bins=int(np.sqrt(N)))
plt.show()
model = build_model(X, Y)
train_model(model, (X, Y), epochs=500)
Xrange = range(X.shape[0])
plt.scatter(Xrange, Y, color="k", label="Data")
y_pred, y_var = model.predict_y(X)

plt.plot(Xrange, y_pred, color="Blue", label="Mean")
plt.plot(Xrange, y_var, label="Var")
upper = y_pred + 1.96 * np.sqrt(y_var)
lower = y_pred - 1.96 * np.sqrt(y_var)
plt.fill_between(Xrange, upper[:, 0], lower[:, 0], alpha=0.3, color="C0", label="CI")
# plt.ylim(0, 15)
plt.legend()
plt.show()
