import tensorflow as tf
import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np

def negatve_log_predictive_density(model, X_test, Y_test, n_samples=100):
        F_samples = model.predict_f_samples(X_test, n_samples)
        # monte-carlazo
        log_pred = model.likelihood.log_prob(X=X_test, F=F_samples, Y=Y_test)
        nlogpred = -tf.reduce_sum(log_pred) / n_samples
        return nlogpred

def mean_squared_error(model, X_test, Y_test):
     Y_mean, _ = model.predict_y(X_test)
     mse = tf.reduce_mean(tf.square(Y_test - Y_mean))
     return mse


def train_model(model, data, epochs=100, log_freq=20, verbose=True):
    """
    Trains the model for a specified number of epochs.
    """
    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.01)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.AdamW(0.01)


    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    for epoch in range(1, epochs + 1):
        optimisation_step()

        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % log_freq == 0 and epoch > 0 and verbose:
            print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")



def plot_gp_predictions(model, X, Y, name):

    Y_mean, Y_var = model.predict_y(X)
    X_range = range(X.shape[0])

    observation_dim = Y.shape[1]
    y_lower = Y_mean - 1.96 * np.sqrt(Y_var)
    y_upper = Y_mean + 1.96 * np.sqrt(Y_var)

    # Calculate the number of rows and columns for the subplot matrix
    n_cols = int(np.ceil(np.sqrt(observation_dim)))
    n_rows = int(np.ceil(observation_dim / n_cols))

    _, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()

    for d in range(observation_dim):
        ax_flat[d].scatter(X_range, Y[:, d], c="k", label="Test Data")
        ax_flat[d].plot(X_range, Y_mean[:, d], label="Predicted Mean")
        ax_flat[d].legend()
        ax_flat[d].fill_between(
        X_range, y_lower[:, d], y_upper[:, d], color="C0", alpha=0.1, label="CI"
        )


    # Hide any unused subplots
    for d in range(observation_dim, n_rows*n_cols):
        ax_flat[d].axis('off')

    plt.tight_layout()
    plt.savefig(name+".png")
    plt.close()