import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from single_output_correlated_chained_gp import SOSVGP, HeteroskedasticLikelihood
from gpflow.utilities import tabulate_module_summary
from sklearn.model_selection import train_test_split
from miscellaneous import *
from datasets import *
from keras.metrics import mean_squared_error, R2Score
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


def train_model(model, data, epochs=500, log_freq=20, validation_split=0.1, patience=5):
    """
    Trains the GP model for a specified number of epochs with early stopping.
    `validation_split` is the proportion of data to be used as validation set.
    `patience` is the number of epochs to wait for improvement before stopping.
    """
    X, Y = data
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split)

    # Training loss function
    train_loss_fn = model.training_loss_closure((X_train, Y_train))
    # Validation loss function
    val_loss_fn = model.training_loss_closure((X_val, Y_val), compile=False)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.001)
    adam_opt = tf.optimizers.Adam(0.01)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    adam_vars = model.trainable_variables

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        natgrad_opt.minimize(train_loss_fn, variational_vars)
        adam_opt.minimize(train_loss_fn, adam_vars)

        if epoch % log_freq == 0 or epoch == 1:
            train_loss = train_loss_fn().numpy()
            val_loss = val_loss_fn().numpy()
            print(f"Epoch {epoch} - Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

            # Check for improvement
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

def evaluate_model(model, x_test, y_test):
    """
    Evaluates the model on the test set and returns the MSE and R2 score.
    """
    Ypred, _ = model.predict_y(x_test)
    r2score = R2Score(class_aggregation=None)

    mse = mean_squared_error(y_test.T, Ypred.numpy().T)
    r2score = r2score(y_test, Ypred.numpy())

    return mse.numpy(), r2score.numpy()

def plot_results(model, x_test, y_test, column_names, P):
    """
    Plots the model predictions and saves the figures.
    """
    Ypred, _ = model.predict_y(x_test)
    Xrange = range(x_test.shape[0])

    for i in range(P):
        plt.figure(figsize=(10, 10))
        plt.plot(Xrange, y_test[:, i], 'k*', label='Testing Data')
        plt.plot(Xrange, Ypred[:, i].numpy(), 'b', label='Predicted')
        plt.title(f'Observed Values {column_names[i]}')
        plt.legend()
        plt.savefig(f"./lmc_results/{column_names[i]}.png")
        plt.close()

def save_metrics_to_csv(column_names, mse, r2score, filename="model_metrics.csv"):
    """
    Saves MSE and R2 score metrics to a CSV file.
    """
    df = pd.DataFrame({'MSE': mse, 'R2 Score': r2score}, index=column_names)
    df.to_csv("./lmc_results/"+filename)

# Main execution
rng = np.random.default_rng(1234)
train_data, test_data, column_names = streamflow_dataset(input_width=1)
x_test, y_test = test_data
x_train, y_train = train_data
n_samples, input_dimension = x_train.shape
_, output_dimension = y_train.shape

N = n_samples
D = input_dimension
M = 25
L = output_dimension
P = output_dimension

random_indexes = rng.choice(range(n_samples), size=M, replace=False)
Zinit = x_train[random_indexes]

kern_list = [gpf.kernels.SquaredExponential(lengthscales=[1]*D) + gpf.kernels.Linear() for _ in range(L)]
kernel = gpf.kernels.LinearCoregionalization(kern_list, W=np.random.randn(P, L))
Zs = [Zinit.copy() for _ in range(P)]
iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
q_mu = np.zeros((M, L))
q_sqrt = np.repeat(np.eye(M)[None, ...], L, axis=0) * 1.0

model = gpf.models.SVGP(kernel, gpf.likelihoods.Gaussian(), inducing_variable=iv, q_mu=q_mu, q_sqrt=q_sqrt)

train_model(model, (x_train, y_train))
mse, r2score = evaluate_model(model, x_test, y_test)

plot_results(model, x_test, y_test, column_names, P)
save_metrics_to_csv(column_names, mse, r2score)
