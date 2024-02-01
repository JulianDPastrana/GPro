#!/home/usuario/Documents/Gpro/gpvenv/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import tensorflow as tf
import tensorflow_probability as tfp
from single_output_correlated_chained_gp import SOSVGP, HeteroskedasticLikelihood
from gpflow.utilities import tabulate_module_summary
from miscellaneous import *
from datasets import *
import warnings
warnings.filterwarnings("ignore")

train_data, test_data, column_names = streamflow_dataset(input_width=1)
x_test, y_test = test_data
x_train, y_train = train_data
n_samples, input_dimension = x_train.shape
_, output_dimension = y_train.shape
# M = 20  # Number of inducing variables for each f_i

# # Initial inducing points position Z
# Z = np.linspace(0, x_train.max(), M)[:, None]  # Z must be of shape [M, 1]
for p in range(output_dimension):
    # Create the CustomSVGP instance
    print(column_names[p])
    # custom_model = SOSVGP(
    #     likelihood_class=tfp.distributions.LogLogistic,
    #     input_dimension=input_dimension,
    #     inducing_points_position=Z.copy()
    #     )

    # # Train the model
    # plt.plot(y_train[:, p][:, None])
    # plt.show()
    # print(np.sum(np.isnan(y_train[:, p][:, None])))
    # custom_model.train_model(train_data=(x_train, y_train[:, p][:, None]), epochs=10)
    data = (x_train, y_train[:, p][:, None])
    # plt.scatter(x_train[:, p][:, None], y_train[:, p][:, None])
    # plt.show()
    # likelihood = gpf.likelihoods.HeteroskedasticTFPConditional(
    # distribution_class=tfp.distributions.LogLogistic,  # Gaussian Likelihood
    # scale_transform=tfp.bijectors.Exp(),  # Exponential Transform
    # )
    likelihood = HeteroskedasticLikelihood(
            distribution_class=tfp.distributions.Weibull,
            param1_transform=tfp.bijectors.Softplus(validate_args=True),
            param2_transform=tfp.bijectors.Softplus(validate_args=True)
        )

    print(f"Likelihood's expected latent_dim: {likelihood.latent_dim}")

    kernel = gpf.kernels.SeparateIndependent(
        [
            gpf.kernels.SquaredExponential(lengthscales=[1]*input_dimension),  # This is k1, the kernel of f1
            gpf.kernels.SquaredExponential(lengthscales=[1]*input_dimension),  # this is k2, the kernel of f2
        ]
    )
    # The number of kernels contained in gpf.kernels.SeparateIndependent must be the same as likelihood.latent_dim
    M = 20  # Number of inducing variables for each f_i

    # Initial inducing points position Z
    # Z = np.random.rand(M, input_dimension)
    Z = x_train[:M, :]
    assert Z.shape[1] == x_train.shape[1] == x_test.shape[1] == input_dimension
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

    # loss_fn = model.training_loss_closure(data)

    # gpf.utilities.set_trainable(model.q_mu, False)
    # gpf.utilities.set_trainable(model.q_sqrt, False)

    # variational_vars = [(model.q_mu, model.q_sqrt)]
    # natgrad_opt = gpf.optimizers.NaturalGradient(gamma=1e-4)

    # adam_vars = model.trainable_variables
    # adam_opt = tf.optimizers.Adam(1e-3)


    # @tf.function
    # def optimisation_step():
    #     natgrad_opt.minimize(loss_fn, variational_vars)
    #     adam_opt.minimize(loss_fn, adam_vars)


    # epochs = 250
    # log_freq = 25

    # for epoch in range(1, epochs + 1):
    #     optimisation_step()

    #     # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
    #     if epoch % log_freq == 0 and epoch > 0:
    #         print(f"Epoch {epoch} - Loss: {loss_fn().numpy() : .4f}")

    def train_generic_model(
        model: gpf.models.GPModel, data: gpf.base.RegressionData
    ) -> None:
        loss = gpf.models.training_loss_closure(model, data)
        opt = gpf.optimizers.Scipy()
        opt.minimize(loss, model.trainable_variables)

    train_generic_model(model, data)

    def mean_squared_error(y, y_pred):
        return np.mean((y - y_pred) ** 2)
    
    def mean_standardized_log_loss(y_true, y_pred, y_std):
        first_term = 0.5 * np.log(2 * np.pi * y_std**2)
        second_term = ((y_true - y_pred)**2)/(2 * y_std**2)
    
        return np.mean(first_term + second_term)
    R2_score = tf.keras.metrics.R2Score()
    # print("Flag")
    # print(model.predict_y(x_test))
    # print(tabulate_module_summary(model))
    pred_mean, pred_var = model.predict_y(x_train)
    
    # print(x_test, x_test.shape)
    print(pred_mean, pred_var)
    print(pred_mean.numpy().max(), pred_var.numpy().max())
    t = np.arange(x_train.shape[0])
    plt.plot(t, pred_mean)
    plt.plot(t, pred_var)
    plt.scatter(t, y_train[:, p][:, None], c="k")
    plt.show()



    pred_mean, pred_var = model.predict_y(x_test)
    
    # print(x_test, x_test.shape)
    print(pred_mean, pred_var)
    print(pred_mean.numpy().max(), pred_var.numpy().max())
    t = np.arange(x_test.shape[0])
    plt.plot(t, pred_mean)
    plt.plot(t, pred_var)
    plt.scatter(t, y_test[:, p][:, None], c="k")
    plt.show()
    mse = mean_squared_error(y_test[:, p][:, None], pred_mean)
    r2 = R2_score(y_test[:, p][:, None], pred_mean.numpy())
    # msll = mean_standardized_log_loss(y_test[:, p][:, None], pred_mean, pred_var)
    print(mse, r2)

    import csv

    with open("test_results.csv", "a") as file:
        writer = csv.DictWriter(file, fieldnames=["Task", "MSE", "R2"])
        writer.writerow({
            "Task": column_names[p],
            "MSE": mse,
            "R2": r2,
            })