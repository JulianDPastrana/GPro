import tensorflow as tf
from gpflow.models import GPModel
from typing import Tuple
from tqdm import tqdm


def continuous_ranked_probability_score(model: GPModel, X_test: tf.Tensor, Y_test: tf.Tensor, n_samples :int = 1500) -> tf.Tensor:
    """
    Computes the Continuous Ranked Probability Score (CRPS) of a GP model.

    Args:
        model (GPModel): The GPflow model.
        X_test (tf.Tensor): The input test data.
        Y_test (tf.Tensor): The true output test data.

    Returns:
        tf.Tensor: The CRPS.
    """
    pass

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


def train_model(model: GPModel, data: Tuple[tf.Tensor, tf.Tensor], batch_size: int = 64, epochs: int = 100) -> None:
    """
    Trains the GP model using minibatch optimization with verbose logging.

    Args:
        model (GPModel): The GPflow model to be trained.
        data (Tuple[tf.Tensor, tf.Tensor]): The training data as a tuple (X, Y).
        batch_size (int): The size of the minibatches. Defaults to 64.
        epochs (int): The number of epochs for training. Defaults to 100.
    """
    X, Y = data

    # Create the dataset and batch it
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(buffer_size=len(X)).batch(batch_size)

    adam_opt = tf.optimizers.Adam(0.01)

    @tf.function
    def optimization_step(batch_X: tf.Tensor, batch_Y: tf.Tensor) -> tf.Tensor:
        with tf.GradientTape() as tape:
            loss = model.training_loss_closure((batch_X, batch_Y), compile=False)()
        gradients = tape.gradient(loss, model.trainable_variables)
        adam_opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        with tqdm(total=len(X) // batch_size, desc=f"Epoch {epoch}/{epochs}", unit="batch") as pbar:
            for batch_X, batch_Y in dataset:
                loss = optimization_step(batch_X, batch_Y)
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix_str(s=f"loss: {epoch_loss / num_batches:.4f}", refresh=True)
                pbar.update(1)
