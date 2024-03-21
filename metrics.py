import tensorflow as tf
import gpflow as gpf

def negatve_log_predictive_density(model, X_test, Y_test, n_samples=100):
        F_samples = model.predict_f_samples(X_test, n_samples)
        # monte-carlazo
        log_pred = model.likelihood.log_prob(X=X_test, F=F_samples, Y=Y_test)
        nlogpred = -tf.reduce_sum(log_pred) / n_samples
        return nlogpred


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