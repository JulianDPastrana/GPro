import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data


def build_model(train_data):
    """
    Builds and returns the GP model.
    """
    X, Y = train_data
    n_samples, input_dim = X.shape
    observation_dim = Y.shape[1]
    latent_dim = 2
    likelihood = LogNormalLikelihood(input_dim, latent_dim, observation_dim)
    kernel = gpf.kernels.SeparateIndependent(
        [gpf.kernels.SquaredExponential(), gpf.kernels.SquaredExponential()]
    )
    print("Obsevation dim: ", likelihood.observation_dim)

    M = 25
    # random_indexes = np.random.choice(range(n_samples), size=M, replace=False)
    # Z = X[random_indexes]
    Z = np.random.randint(0, 1, size=(M, 1))

    inducing_variable = gpf.inducing_variables.SeparateIndependentInducingVariables(
        [gpf.inducing_variables.InducingPoints(Z), gpf.inducing_variables.InducingPoints(Z)]
    )

    # initialize mean of variational posterior to be of shape MxL
    q_mu = np.zeros((M, 2))
    # initialize \sqrt(Σ) of variational posterior to be of shape LxMxM
    q_sqrt = np.repeat(np.eye(M)[None, ...], 2, axis=0) * 1.0

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    return model

def train_model(model, data, epochs=100, log_freq=20):
    """
    Trains the model for a specified number of epochs.
    """
    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.01)

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


def main():
    train_data, test_data = get_uv_data()
    X_train, Y_train = test_data
    X_test, Y_test = test_data

    model = build_model(train_data)
    train_model(model, train_data, epochs=500)
    
    Y_mean, Y_var = model.predict_y(X_test)
    X_range = range(X_test.shape[0])

    fig, ax = plt.subplots()
    ax.scatter(X_range, Y_test, c="k")
    ax.plot(X_range, Y_mean)
   
    y_lower = Y_mean - 1.96 * np.sqrt(Y_var)
    y_upper = Y_mean + 1.96 * np.sqrt(Y_var)

    ax.fill_between(
        X_range, y_lower[:, 0], y_upper[:, 0], color="C0", alpha=0.1
    )
    plt.show()
    

if __name__ == "__main__": 
    main()