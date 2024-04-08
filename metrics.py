import tensorflow as tf
import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
from likelihoods import LogNormalLikelihood
from gpflow.quadrature import ndiag_mc, ndiagquad, mvnquad

def negatve_log_predictive_density(model, X_test, Y_test, n_samples=500):
        F_samples = model.predict_f_samples(X_test, n_samples)
        # monte-carlazo
        log_pred = model.likelihood.log_prob(X=X_test, F=F_samples, Y=Y_test)
        nlogpred = -tf.reduce_sum(log_pred) / n_samples
        
        return nlogpred

def mean_squared_error(model, X_test, Y_test):
    Y_mean, _ = model.predict_y(X_test)
    print("Anoter NaNs", np.sum(~np.isfinite(Y_mean.numpy())))
    print("Anoter NaNs but test", np.sum(~np.isfinite(Y_test)))
    mse = tf.reduce_mean(tf.square(Y_test - Y_mean))
    print("Anoter NaNs but mse", np.sum(~np.isfinite(mse.numpy())))
    return mse

def mean_absolute_error(model, X_test, Y_test):
     Y_mean, _ = model.predict_y(X_test)
     mae = tf.reduce_mean(tf.math.abs(Y_test - Y_mean))
     return mae


def train_model(model, data, validation_data, epochs=100, log_freq=20, patience=10, verbose=True):
    """
    Trains the model for a specified number of epochs with early stopping and saves the best model.
    
    Parameters:
    - model: The GPflow model to train.
    - data: The training data.
    - validation_data: A subset of data for validation.
    - epochs: The maximum number of epochs to train for.
    - log_freq: The frequency with which to log the training progress.
    - patience: The number of epochs to wait for improvement on the validation set before stopping.
    - verbose: Whether to print progress messages.
    """
    loss_fn = model.training_loss_closure(data, compile=True)
    val_loss_fn = model.training_loss_closure(validation_data, compile=True)
    # gpf.utilities.set_trainable(model.q_mu, False)
    # gpf.utilities.set_trainable(model.q_sqrt, False)
    # variational_vars = [(model.q_mu, model.q_sqrt)]
    # natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.005)
    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.005)
    grads, variables = zip(*adam_opt.compute_gradients(loss_fn, adam_vars))
    grads, _ = tf.clip_by_global_norm(grads, 1e3)
    adam_opt.apply_gradients(zip(grads, variables))

    # Setup checkpointing
    checkpoint = tf.train.Checkpoint(optimizer=adam_opt, model=model)
    manager = tf.train.CheckpointManager(checkpoint, './tf_ckpts', max_to_keep=1)

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    @tf.function
    def optimisation_step():
        adam_opt.minimize(loss_fn, adam_vars)
        # natgrad_opt.minimize(loss_fn, variational_vars)

    for epoch in range(1, epochs + 1):
        optimisation_step()

        val_loss = val_loss_fn().numpy()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the model as it's the best so far
            save_path = manager.save()
            if False:
                print(f"Saved checkpoint for epoch {epoch}: {save_path}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
            break

        if epoch % log_freq == 0 and verbose:
            train_loss = loss_fn().numpy()
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    # Optionally, restore the best model
    checkpoint.restore(manager.latest_checkpoint)
    if verbose and manager.latest_checkpoint:
        print(f"Restored best model from {manager.latest_checkpoint}")


def create_independent_model(observation_dim, input_dim, num_inducing):
    Zinit = np.random.rand(num_inducing, input_dim)
    
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(observation_dim)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    iv_list = [gpf.inducing_variables.InducingPoints(Zinit) for _ in range(observation_dim)]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    
    return gpf.models.SVGP(
        kernel=kernel,
        likelihood=gpf.likelihoods.Gaussian(),
        inducing_variable=iv,
        num_latent_gps=observation_dim
    )

def create_lmc_model(observation_dim, input_dim, num_inducing, indfun_dim):
    Zinit = np.random.rand(num_inducing, input_dim)
    
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(indfun_dim)]
    kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(observation_dim, indfun_dim))
    iv_list = [gpf.inducing_variables.InducingPoints(Zinit) for _ in range(indfun_dim)]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    q_mu = np.zeros((num_inducing, indfun_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], indfun_dim, axis=0) * 1.0
    return gpf.models.SVGP(
        kernel=kernel,
        likelihood=gpf.likelihoods.Gaussian(),
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

def create_chained_independent_model(observation_dim, input_dim, num_inducing):
    latent_dim = 2 * observation_dim 
    Zinit = np.random.rand(num_inducing, input_dim)
    
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(latent_dim)]
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    iv_list = [gpf.inducing_variables.InducingPoints(Zinit) for _ in range(latent_dim)]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    
    return gpf.models.SVGP(
        kernel=kernel,
        likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
        inducing_variable=iv,
        num_latent_gps=observation_dim
    )

def create_chained_correlated_model(observation_dim, input_dim, num_inducing, indfun_dim):
    latent_dim = 2 * observation_dim
    Zinit = np.random.rand(num_inducing, input_dim)
    
    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) for _ in range(indfun_dim)]
    kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(latent_dim, indfun_dim))
    iv_list = [gpf.inducing_variables.InducingPoints(Zinit) for _ in range(indfun_dim)]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    q_mu = np.zeros((num_inducing, indfun_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], indfun_dim, axis=0) * 1.0
    return gpf.models.SVGP(
        kernel=kernel,
        likelihood=LogNormalLikelihood(input_dim, latent_dim, observation_dim),
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )


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


def plot_results(model, X, Y):
    n_samples = 100
    n_y_pixels = 100
    N = X.shape[0]
    X_range = range(N)
    observation_dim = Y.shape[1]
    # Calculate the number of rows and columns for the subplot matrix
    n_cols = int(np.ceil(np.sqrt(observation_dim)))
    n_rows = int(np.ceil(observation_dim / n_cols))

    _, ax = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))

    # Flatten the ax array for easy indexing
    ax_flat = ax.flatten()
    res = np.zeros((N, n_y_pixels, observation_dim))
    F_samples = model.predict_f_samples(X, n_samples)
    for j in range(N):
         for i in range(n_samples):
            F_sample = F_samples[i, j, :].reshape(1, -1)
            lin_spaced = np.linspace(Y.min(), Y.max(), n_y_pixels)[:, None]
            print(F_sample.shape)
            prob = np.exp(model.likelihood.log_prob(X=X, F=F_sample, Y=tiled))
            print(prob.shape)
            res[j,:,:] += prob.numpy
