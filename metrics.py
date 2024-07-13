import tensorflow as tf
import gpflow as gpf
import matplotlib.pyplot as plt
import numpy as np
import tensorflow_probability as tfp
from typing import Optional
from gpflow.models import GPModel
from typing import Tuple
from tqdm import tqdm
from likelihoods import MOChainedLikelihoodMC
import properscoring as ps

def continuous_ranked_probability_score_gaussian(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor) -> tf.Tensor:
    """
    continuous_ranked_probability_score
    """
    Y_mean, Y_var = model.predict_y(X_test)
    return ps.crps_gaussian(Y_test, Y_mean, Y_var).mean()

def mean_standardized_log_loss(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor, Y_train: tf.Tensor) -> tf.Tensor:
    """
    Computes the Mean Standardized Log Loss (MSLL) of a GP model
        Args:
        model (GPModel): The GPflow model.
        X_test (tf.Tensor): The input test data.
        Y_test (tf.Tensor): The true output test data.
        Y_train (tf.Tensor): The true output train data.

    Returns:
        tf.Tensor: The negative log predictive density.
    """
    Y_mean, Y_var = model.predict_y(X_test)
    model_nlp = (0.5 * np.log(2 * np.pi * Y_var)
                 + 0.5 * (Y_test - Y_mean) ** 2 / Y_var)
    mu, sig = Y_train.mean(), Y_train.var()
    data_nlp = (0.5 * np.log(2 * np.pi * sig)
                 + 0.5 * (Y_test - mu) ** 2 / sig)
    loss = np.mean(model_nlp - data_nlp)
    return loss


def negative_log_predictive_density(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor) -> tf.Tensor:
    """
    Computes the negative log predictive density (NLPD) of a GP model.

    Args:
        model (GPModel): The GPflow model.
        X_test (tf.Tensor): The input test data.
        Y_test (tf.Tensor): The true output test data.

    Returns:
        tf.Tensor: The negative log predictive density.
    """
    Fmu, Fvar = model.predict_f(X_test)
    lpd = model.likelihood.predict_log_density(X=X_test, Fmu=Fmu, Fvar=Fvar, Y=Y_test)
    return -tf.reduce_mean(lpd)


def mean_squared_error(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor) -> tf.Tensor:
    """
    Computes the mean squared error (MSE) between predicted and true outputs.

    Args:
        model (GPModel): The GPflow model.
        X_test (tf.Tensor): The input test data.
        Y_test (tf.Tensor): The true output test data.

    Returns:
        tf.Tensor: The mean squared error.
    """
    Y_mean, _ = model.predict_y(X_test)
    mse = tf.reduce_mean(tf.square(Y_test - Y_mean))
    return mse


def mean_absolute_error(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor) -> tf.Tensor:
    """
    Computes the mean absolute error (MAE) between predicted and true outputs.

    Args:
        model (GPModel): The GPflow model.
        X_test (tf.Tensor): The input test data.
        Y_test (tf.Tensor): The true output test data.

    Returns:
        tf.Tensor: The mean absolute error.
    """
    Y_mean, _ = model.predict_y(X_test)
    mae = tf.reduce_mean(tf.math.abs(Y_test - Y_mean))
    return mae


def train_model(model, data: Tuple[tf.Tensor, tf.Tensor], batch_size: int = 64, epochs: int = 100, patience: int = 3) -> None:
    """
    Trains the GP model using minibatch optimization with verbose logging and early stopping based on training loss.
    Restores the model to the best state observed during training.

    Args:
        model: The GPflow model to be trained.
        data (Tuple[tf.Tensor, tf.Tensor]): The training data as a tuple (X, Y).
        batch_size (int): The size of the minibatches. Defaults to 64.
        epochs (int): The number of epochs for training. Defaults to 100.
        patience (int): The number of epochs to wait for an improvement in training loss before stopping early. Defaults to 3.
    """
    X, Y = data

    # Create the dataset and batch it
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)

    adam_opt = tf.optimizers.Adamax(0.1)

    @tf.function
    def optimization_step(batch_X: tf.Tensor, batch_Y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = model.training_loss_closure((batch_X, batch_Y), compile=False)()
        gradients = tape.gradient(loss, model.trainable_variables)
        adam_opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    # Early stopping variables
    best_train_loss = np.inf
    epochs_without_improvement = 0
    # best_parameters = gpf.utilities.parameter_dict(model)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        with tqdm(total=len(X) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for batch_X, batch_Y in dataset:
                loss = optimization_step(batch_X, batch_Y)
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix_str(s=f"loss: {epoch_loss / num_batches:.4e}", refresh=True)
                pbar.update(1)

        # Calculate average training loss for the epoch
        avg_epoch_loss = epoch_loss / num_batches

        # Check for improvement
        if avg_epoch_loss < best_train_loss:
            best_train_loss = avg_epoch_loss
            epochs_without_improvement = 0
            best_epoch = epoch
            # Save the best parameters
            # best_parameters = gpf.utilities.parameter_dict(model)
            log_dir = "./checkpoints"
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, log_dir, max_to_keep=3)
            manager.save()
        else:
            epochs_without_improvement += 1

        # Print training loss
        print(f"Epoch {epoch}: Training loss: {avg_epoch_loss:.4e}")

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best training loss: {best_train_loss:.4e} at epoch {best_epoch}")
            # gpf.utilities.multiple_assign(model, best_parameters)
            ckpt.restore(manager.latest_checkpoint)

            break
        
    ckpt.restore(manager.latest_checkpoint)

def plot_confidence_interval(
    y_mean: np.ndarray, 
    y_var: np.ndarray, 
    task_name: str, 
    fname: Optional[str] = None, 
    y_true: Optional[np.ndarray] = None
) -> None:
    """
    Plot the predictive mean and confidence intervals for a given task.

    Parameters:
        y_mean (np.ndarray): The predictive mean values.
        y_var (np.ndarray): The predictive variance values.
        task_name (str): The name of the task for labeling the plot.
        fname (Optional[str]): The filename to save the plot. If None, the plot is displayed.
        y_true (Optional[np.ndarray]): The true values to be plotted for comparison. If None, they are not plotted.

    Returns:
        None
    """
    time_range = range(len(y_mean))

    # Plot the confidence intervals
    for k in range(1, 3):
        lb = y_mean - k * np.sqrt(y_var)
        ub = y_mean + k * np.sqrt(y_var)
        plt.fill_between(time_range, lb, ub, color="silver", alpha=1 - 0.05 * k ** 3, label=f'$\mp${k}$\sigma$ interval')

    # Plot the predictive mean
    plt.plot(time_range, y_mean, color="black", label="Predictive Mean")

    # Plot the true values if provided
    if y_true is not None:
        plt.scatter(time_range, y_true, color="red", alpha=0.8, label="True Values")

    # Add title and labels
    plt.title(f"Output {task_name}")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()

    # Save or show the plot
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

    # Close the plot to free resources
    plt.close()