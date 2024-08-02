import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow as gpf
import tensorflow_probability as tfp
import pickle
import seaborn as sns
from gpflow.models import GPModel
from data_exploration import WindowGenerator
from typing import Tuple, Optional
from likelihoods import MOChainedLikelihoodMC
import properscoring as ps
import matplotlib.pyplot as plt
import tikzplotlib as tikz

def plot_predict_log_density(
    model,
    X,
    Y,
    task,
    path
):
    """
    Plot the predictive density of a model's forecasts along with the observed data.

    Parameters:
    model (Any): The trained model that provides predictive distribution and likelihood functions.
    X (np.ndarray): The input features of the dataset, typically time indices in forecast scenarios.
    Y (np.ndarray): The target values of the dataset, typically quantities like power generation.
    task (int): The index of the task for which the model is trained.

    The function saves the plot as 'predictive_density.png' in the current directory.

    The plotting includes the observation points, the mean of the forecast, and the
    contour fill of the predictive density distribution.
    """
    
    # Initialize the time index for X-axis
    time_index = np.arange(X.shape[0])[:, None]
    # Define constants for the number of samples and Y-axis pixels
    num_samples = 15
    num_y_pixels = 50
    observation_dim = Y.shape[1]
    # Calculate X and Y limits for the plot
    xlim_min, xlim_max = time_index[:, 0].min(), time_index[:, 0].max()
    ylim_min, ylim_max = 0, 1
    
    # Sample from the model's predictive distribution
    Fsamples = model.predict_f_samples(X, num_samples)
    
    # Transform samples using the likelihood parameters
    # num_samples x time_index x observation_dim
    alpha = model.likelihood.param1_transform(Fsamples[..., ::2])
    beta = model.likelihood.param1_transform(Fsamples[..., 1::2])
    
    # Prepare line space for the Y-axis
    line_space = np.linspace(ylim_min, ylim_max, num_y_pixels)
    predictive_density = np.zeros((X.shape[0], num_y_pixels))
    # Compute the predictive density for the specified task
    # dist = model.likelihood.distribution_class(alpha[:, :, task], beta[:, :, task])
    for i in range(num_samples):
        for j in range(X.shape[0]):
            dist = model.likelihood.distribution_class(
                alpha[i, j , task],
                beta[i, j, task],
                force_probs_to_zero_outside_support=True,
            )
            predictive_density[j] += dist.prob(line_space).numpy()
    predictive_density /= num_samples

    
    # Create a meshgrid for the contour plot
    x_mesh, y_mesh = np.meshgrid(time_index, line_space)

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    cs = ax.contourf(x_mesh, y_mesh, np.log10(predictive_density.T+1e-12))
    ax.scatter(time_index, Y[:, task], color="k", s=10, label="Observaciones")

    plt.colorbar(cs, ax=ax)
    plt.tight_layout()
    plt.savefig(path + f"/predictive_density_{task}.png")
    plt.close()
            


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



def train_model(model, data: Tuple[tf.Tensor, tf.Tensor], epochs: int = 5000, patience: int = 150) -> None:
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

    loss_fn = model.training_loss_closure(data)

    gpf.utilities.set_trainable(model.q_mu, False)
    gpf.utilities.set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = gpf.optimizers.NaturalGradient(gamma=0.1)
    # print(natgrad_opt.xi_transform)
    # assert 0==1

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.01)
    
    @tf.function
    def optimisation_step():
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    # Early stopping variables
    best_train_loss = np.inf
    epochs_without_improvement = 0


    for epoch in range(1, epochs + 1):
        optimisation_step()
        epoch_loss = loss_fn().numpy()
        # For every 'log_freq' epochs, print the epoch and plot the predictions against the data
        if epoch % 50 == 0 and epoch > 0:
            print(f"Epoch {epoch} - Loss: {epoch_loss :.4e}")

        # Calculate average training loss for the epoch
        avg_epoch_loss = epoch_loss 

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


    

        # Early stopping check
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best training loss: {best_train_loss:.4e} at epoch {best_epoch}")
            # gpf.utilities.multiple_assign(model, best_parameters)
            ckpt.restore(manager.latest_checkpoint)

            break
            
    ckpt.restore(manager.latest_checkpoint)

def load_datasets():
    dataset = pd.read_excel(
        io='./Ejercicio hidrología v2.xlsx',
        header=[0, 1],
        index_col=0,
        sheet_name='Datos generales'
    ).droplevel(level=1, axis=1)
    dataset.fillna(0.0, inplace=True)
    # uv_data = dataset.iloc[:, 3::5]
    ct_data = dataset.iloc[:4410, 1::5]

    # num_uv = uv_data._get_numeric_data()
    num_ct = ct_data._get_numeric_data()
    # num_uv[num_uv < 0] = 0
    num_ct[num_ct < 0] = 0

    return ct_data

def normalize_dataset(dataframe: pd.DataFrame) -> tuple:
    mean = dataframe.mean()
    std = dataframe.std()
    norm_dataset = (dataframe - mean) / std
    norm_dataset.fillna(0.0, inplace=True)

    return norm_dataset, mean, std


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
    plt.figure(figsize=(16, 8))
    time_range = range(len(y_mean))

    # Plot the confidence intervals
    lb = y_mean - 2 * np.sqrt(y_var)
    ub = y_mean + 2 * np.sqrt(y_var)
    plt.fill_between(time_range, lb, ub, color="b", alpha=0.5, label=f'2 $\sigma$ interval')

    # Plot the predictive mean
    plt.plot(time_range, y_mean, ".b-", label="Predictive Mean")

    # Plot the true values if provided
    if y_true is not None:
        plt.plot(time_range, y_true, 'r.-', label="True Values")

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


def ind_gp(input_dim, observation_dim, num_inducing, X_train):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [
        gpf.kernels.SquaredExponential(
            lengthscales=np.random.uniform(0.01, np.log(input_dim)*np.sqrt(input_dim), size=input_dim)) for _ in range(observation_dim)
        ]
        
    kernel = gpf.kernels.SeparateIndependent(kern_list)
    
    Zinit = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(num_inducing, input_dim))
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Zinit)
        )

    
    likelihood = gpf.likelihoods.Gaussian(variance=np.ones(shape=observation_dim))

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        num_latent_gps=observation_dim
    )

    return model

def lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train):
    # Create a list of base kernels for the Linear Coregionalization model
    kern_list = [
        gpf.kernels.SquaredExponential(
            lengthscales=np.random.uniform(0.01, np.log(input_dim)*np.sqrt(input_dim), size=input_dim)) for _ in range(ind_process_dim)
        ]
    W = np.random.randn(observation_dim, ind_process_dim) * 0.1 #+ np.eye(observation_dim, ind_process_dim)
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=W
    )
    
    Zinit = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(num_inducing, input_dim))
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Zinit)
        )

    
    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    
    likelihood = gpf.likelihoods.Gaussian(variance=np.ones(shape=observation_dim))

    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    for q in range(ind_process_dim):
        gpf.utilities.set_trainable(model.kernel.kernels[q].variance, False)

 
    return model


def chd_lmc_gp(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing, X_train):
    kern_list = [
        gpf.kernels.SquaredExponential(
            lengthscales=np.random.uniform(0.01, np.log(input_dim)*np.sqrt(input_dim), size=input_dim)) for _ in range(ind_process_dim)
        ]
    # W = np.random.uniform(-0.25, 0.25, size=(latent_dim, ind_process_dim))
    
    # assert 0 == 1
    # * 0.01 * np.log(latent_dim)*np.sqrt(ind_process_dim)
    # W = np.random.uniform(0.01, np.log(latent_dim)*np.sqrt(ind_process_dim), size=(latent_dim, ind_process_dim))
    
    W = np.random.randn(latent_dim, ind_process_dim) * 0.1 / 2

    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=W
    )

    Zinit = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(num_inducing, input_dim))
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Zinit)
        )


    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    
    likelihood = MOChainedLikelihoodMC(
            input_dim=input_dim,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            distribution_class=tfp.distributions.Normal,
            param1_transform=lambda x: x,
            param2_transform=tf.math.softplus
        )
    
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    return model


def chd_ind_gp(input_dim, latent_dim, observation_dim, num_inducing, X_train):
    kern_list = [
        gpf.kernels.SquaredExponential(
            lengthscales=np.random.uniform(0.01, np.log(input_dim)*np.sqrt(input_dim), size=input_dim)) for _ in range(latent_dim)
        ]
        
    kernel = gpf.kernels.SeparateIndependent(kern_list)

    Zinit = np.random.uniform(X_train.min(axis=0), X_train.max(axis=0), size=(num_inducing, input_dim))
    iv = gpf.inducing_variables.SharedIndependentInducingVariables(
            gpf.inducing_variables.InducingPoints(Zinit)
        )

    
    likelihood = MOChainedLikelihoodMC(
            input_dim=input_dim,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            distribution_class=tfp.distributions.Gamma,
            param1_transform=tf.math.softplus,
            param2_transform=tf.math.softplus
        )
    
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        num_latent_gps=latent_dim
    )

    return model

def dump_load_model(
        path,
        model_name,
        model,
        train_data
):

    try:
        with open(path + model_name, 'rb') as file:
            params = pickle.load(file)
        gpf.utilities.multiple_assign(model, params)
        print("Model parameters loaded successfully.")

    except FileNotFoundError:
        train_model(model, data=train_data)
        with open(path + model_name, 'wb') as handle:
            pickle.dump(gpf.utilities.parameter_dict(model), handle)
        print("Model trained and parameters saved successfully.")


    

def ind_model():
    path = "./ind_tests"
    model = ind_gp(input_dim, observation_dim, num_inducing, X_train)
    model_name = f"/indgp_Normal_T{order}_M{num_inducing}.pkl"

    dump_load_model(path, model_name, model)
        

    nlpd = negative_log_predictive_density(model, X_test, Y_test)
    msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
    crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)

    print(f"NLPD: {nlpd}")
    print(f"MSLL: {msll}")
    print(f"CRPS: {crps}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    Y_mean, Y_var = model.predict_y(X_test)

    for task in range(observation_dim):
        plot_confidence_interval(
            y_mean=Y_mean[:, task],
            y_var=Y_var[:, task],
            task_name=f"task_{task}",
            fname=path+f"/task_{task}",
            y_true=Y_test[:, task]
        )

def plot_lengthscales(model, ind_process_dim, path, filename):
    plt.figure(figsize=(16, 8))
    lengthscale_matrix = np.empty(shape=(ind_process_dim, input_dim))
    for q in range(ind_process_dim):
        lengthscale_matrix[q] = model.kernel.kernels[q].lengthscales.numpy()

    sns.heatmap(
            lengthscale_matrix,
            cbar=True,
            cmap="viridis",
            vmin=10,
            vmax=40,
            )

    tikz.save(path + filename + ".tex")
    plt.savefig(path + filename + ".png")
    plt.close()


def lmc_model():
    
    path = "./lmc_tests"
    filename = "/lmc_grid_Q"
    results_df = pd.DataFrame()
    for q in [17]:#in range(1, 2*observation_dim + 1):
        ind_process_dim = q
        model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)
        model_name = f"/lmcgp_Normal_T{order}_M{num_inducing}_Q{ind_process_dim}.pkl"
        
        dump_load_model(path, model_name, model)
        plot_lengthscales(
                model=model,
                ind_process_dim=ind_process_dim,
                path=path
                )
        plt.figure(figsize=(16, 8))
        sns.heatmap(
                np.abs(model.kernel.W.numpy()),
                cbar=True,
                cmap="viridis"
                )
        tikz.save(path + f"/coregionalization_{ind_process_dim}.tex")
        plt.savefig(path + f"/coregionalization_{ind_process_dim}.png")
        plt.close()
       
        """""
        nlpd = negative_log_predictive_density(model, X_test, Y_test)
        msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
        crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
        mse = mean_squared_error(model, X_test, Y_test)
        mae = mean_absolute_error(model, X_test, Y_test)

        results_df.loc[q, "NLPD"] = nlpd.numpy()
        results_df.loc[q, "MSLL"] = msll
        results_df.loc[q, "CRPS"] = crps
        results_df.loc[q, "MSE"] = mse.numpy()
        results_df.loc[q, "MAE"] = mae.numpy()

       # Save results to Excel file
        try:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='a', if_sheet_exists="replace") as writer:
                results_df.to_excel(writer)
        except FileNotFoundError:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='w') as writer:
                results_df.to_excel(writer)

    df = pd.read_excel(path + filename + ".xlsx")
    metrics = ["MSLL", "CRPS", "MSE", "NLPD"]

    for metric in metrics:
        plt.figure(figsize=(16, 8))
        plt.plot(df["q"], df[metric], marker='o')
        plt.xlabel("Number of Independent GPs (Q)")
        plt.ylabel(f"{metric} Value")
        plt.grid(True)
        tikz.save(path + f"/{metric}_gs.tex")
        plt.savefig(path + f"/{metric}_gs.png")
        plt.close()
    """""

def lmcpre_model():
    path = "./lmc_tests/pretrained_models"
    model_w = lmc_gp(input_dim, observation_dim, observation_dim, num_inducing, X_train)

    model_name_w = f"/lmcgp_Normal_T{order}_Q{observation_dim}_M{num_inducing}_pretrained.pkl"
    with open(f"ind_tests/indgp_Normal_T{order}_M{num_inducing}.pkl", 'rb') as file:
        params_ind = pickle.load(file)
    gpf.utilities.multiple_assign(model_w, params_ind)

    model_w.kernel.W.assign(np.eye(observation_dim, observation_dim))

    gpf.utilities.set_trainable(model_w, False)
    gpf.utilities.set_trainable(model_w.kernel.W, True)
    train_model(model_w, data=train_data)
    gpf.utilities.set_trainable(model_w, False)
    gpf.utilities.set_trainable(model_w.likelihood, True)
    train_model(model_w, data=train_data)

    with open(path + model_name_w, 'wb') as handle:
        pickle.dump(gpf.utilities.parameter_dict(model_w), handle)
    print("Model trained and parameters saved successfully.")

    with open(path + model_name_w, 'rb') as file:
        params = pickle.load(file)
    gpf.utilities.multiple_assign(model_w, params)
    print("Model parameters loaded successfully.")

    nlpd = negative_log_predictive_density(model_w, X_test, Y_test)
    msll = mean_standardized_log_loss(model_w, X_test, Y_test, Y_train)
    crps = continuous_ranked_probability_score_gaussian(model_w, X_test, Y_test)
    mse = mean_squared_error(model_w, X_test, Y_test)
    mae = mean_absolute_error(model_w, X_test, Y_test)

    print(f"NLPD: {nlpd}")
    print(f"MSLL: {msll}")
    print(f"CRPS: {crps}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    
    W = model_w.kernel.W.numpy()
    rq = np.sum(W**2, axis=0)

    sort_idx = np.argsort(rq)[::-1]

    numbers = np.arange(observation_dim)
    plt.bar(numbers, rq[sort_idx])
    plt.xticks(ticks=numbers, labels=numbers[sort_idx])
    plt.savefig(path + "/relevance.png")

    W_sort = W[:, sort_idx]
    X_train_sort = X_train[:, sort_idx]
    X_test_sort = X_test[:, sort_idx]
    results_df = pd.DataFrame()
    filename = "/lmc_relevance"
    for q in range(1, observation_dim+1):
        print(f"q: {q}")
        X_trainq = X_train# X_train_sort[:, :q]
        X_testq = X_test# X_test_sort[:, :q]
        _, input_dimq = X_trainq.shape
        print(X_trainq.shape, X_testq.shape)
        model = lmc_gp(input_dimq, observation_dim, q, num_inducing, X_trainq)
        

        # for i in range(q):
        #     model.kernel.kernels[i].variance.assign(params[f".kernel.kernels[{sort_idx[i]}].variance"]) 
        #     model.kernel.kernels[i].lengthscales.assign(params[f".kernel.kernels[{sort_idx[i]}].lengthscales"].numpy()[sort_idx][:q])

        # model.inducing_variable.inducing_variable.Z.assign(params[f".inducing_variable.inducing_variable.Z"].numpy()[:, sort_idx][:, :i+1])
        # model.q_mu.assign(params[f".q_mu"].numpy()[:, sort_idx][:, :i+1])
        # model.q_sqrt.assign(params[f".q_sqrt"].numpy()[sort_idx][:i+1])
        # model.likelihood.variance.assign(params[".likelihood.variance"])

        model.kernel.W.assign(W_sort[:, :q])    
        model_name = f"/lmc_gp_q{q}.pkl"
         
        try:
            with open(path + model_name, 'rb') as file:
                params_ = pickle.load(file)
            gpf.utilities.multiple_assign(model, params_)
            print("Model parameters loaded successfully.")
        
        except FileNotFoundError:    
        
            train_model(model, (X_trainq, Y_train))
            with open(path + model_name, 'wb') as handle:
                pickle.dump(gpf.utilities.parameter_dict(model), handle)
            print("Model trained and parameters saved successfully.")

        nlpd = negative_log_predictive_density(model, X_testq, Y_test)
        msll = mean_standardized_log_loss(model, X_testq, Y_test, Y_train)
        crps = continuous_ranked_probability_score_gaussian(model, X_testq, Y_test)
        mse = mean_squared_error(model, X_testq, Y_test)
        mae = mean_absolute_error(model, X_testq, Y_test)


        results_df.loc[q, "NLPD"] = nlpd.numpy()
        results_df.loc[q, "MSLL"] = msll
        results_df.loc[q, "CRPS"] = crps
        results_df.loc[q, "MSE"] = mse.numpy()
        results_df.loc[q, "MAE"] = mae.numpy()

        # Save results to Excel file
        try:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='a', if_sheet_exists="replace") as writer:
                results_df.to_excel(writer)
        except FileNotFoundError:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='w') as writer:
                results_df.to_excel(writer)


def chd_ind_model():
    # path = "./chd_ind_tests"
    path = "./chd_ind_lognormal"
    latent_dim = 2 * observation_dim
    model = chd_ind_gp(input_dim, latent_dim, observation_dim, num_inducing, X_train)
    # model_name = f"/chdindgp_Normal_T{order}_M{num_inducing}_softplus.pkl"
    model_name = f"/chdindgp_LogNormal_T{order}_M{num_inducing}.pkl"
    dump_load_model(path, model_name, model)
        

    nlpd = negative_log_predictive_density(model, X_test, Y_test)
    msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
    crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)

    print(f"NLPD: {nlpd}")
    print(f"MSLL: {msll}")
    print(f"CRPS: {crps}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    for task in range(observation_dim):
        plot_predict_log_density(
            model,
            X_test,
            Y_test,
            task=task,
            path=path
        )
    # Y_mean, Y_var = model.predict_y(X_test)

    # for task in range(observation_dim):
    # plot_confidence_interval(
    # y_mean=Y_mean[:, task],
    # y_var=Y_var[:, task],
    # task_name=f"task_{task}",
    # fname=path+f"/task_{task}",
    # y_true=Y_test[:, task]
    # )


def chd_corr_model():
    path = "./chd_tests"
    # path = "./chd_lmc_gamma"
    filename = "/chd_grid_Q_Normal"
    results_df = pd.DataFrame()
    latent_dim = 2 * observation_dim
    for q in range(36, 2*observation_dim+1):
        # break
        print(f"q: {q}")
        ind_process_dim = q
        model = chd_lmc_gp(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing, X_train)
        model_name = f"/chdgp_Normal_T{order}_M{num_inducing}_Q{ind_process_dim}_Normal.pkl"
        # model_name = f"/chd_Gamma_T{order}_M{num_inducing}_Q{ind_process_dim}.pkl"

        dump_load_model(path, model_name, model)


        plt.figure(figsize=(16, 8))
        sns.heatmap(model.kernel.W.numpy(), annot=True, fmt=".2f")
        plt.savefig(path + f"/coregionalization_{ind_process_dim}.png")
        plt.close()

        nlpd = negative_log_predictive_density(model, X_test, Y_test)
        msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
        crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
        mse = mean_squared_error(model, X_test, Y_test)
        mae = mean_absolute_error(model, X_test, Y_test)

        results_df.loc[q, "NLPD"] = nlpd.numpy()
        results_df.loc[q, "MSLL"] = msll
        results_df.loc[q, "CRPS"] = crps
        results_df.loc[q, "MSE"] = mse.numpy()
        results_df.loc[q, "MAE"] = mae.numpy()

       # Save results to Excel file
        try:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='a', if_sheet_exists="replace") as writer:
                results_df.to_excel(writer)
        except FileNotFoundError:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='w') as writer:
                results_df.to_excel(writer)

    ind_process_dim = 5
    model = chd_lmc_gp(input_dim, latent_dim, observation_dim, ind_process_dim, num_inducing, X_train)
    model_name = f"/chdgp_Normal_T{order}_M{num_inducing}_Q{ind_process_dim}_Normal.pkl"

    with open(f"chd_tests"+model_name, 'rb') as file:
         params_ind = pickle.load(file)
    gpf.utilities.multiple_assign(model, params_ind)

    dump_load_model(path, model_name, model)
        

    nlpd = negative_log_predictive_density(model, X_test, Y_test)
    msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
    crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)

    print(f"NLPD: {nlpd}")
    print(f"MSLL: {msll}")
    print(f"CRPS: {crps}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    Y_mean, Y_var = model.predict_y(X_test)

    for task in range(observation_dim):
        plot_confidence_interval(
             y_mean=Y_mean[:, task],
             y_var=Y_var[:, task],
             task_name=f"task_{task}",
             fname=path+f"/task_{task}",
             y_true=Y_test[:, task]
         )



def train_lmc_by_horizon():

    path = "./lmc_tests/across_horizons/"
    filename = "/lmc_grid_H" 
    results_df = pd.DataFrame()
    np.random.seed(1966)
    tf.random.set_seed(1966)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    ct_data = load_datasets()
    indexes = [2, 9, 11, 12, 13, 14, 16, 17, 19, 21, 3, 6, 0, 5, 7, 8, 10, 1, 4, 18, 22, 20, 15]
    ct_data = ct_data.iloc[:, indexes]

    norm_data, data_mean, data_std = normalize_dataset(ct_data)
    
    horizon_list = [6] # [1, 2, 3, 4, 5, 6, 7, 14, 21, 30]
    for horizon in horizon_list: 
        order = 1
        input_width = order
        label_width = 1
        shift = horizon
        window = WindowGenerator(
            input_width,
            label_width,
            shift,
            norm_data.columns
        )
        x, y = window.make_dataset(norm_data)
        N = len(x)
        X_train, Y_train = x[0:int(N * 0.9)], y[0:int(N * 0.9)]
        X_test, Y_test = x[int(N * 0.9):], y[int(N * 0.9):]
        train_data = (X_test, Y_test)
        
        global input_dim

        _, input_dim = X_train.shape
        observation_dim = Y_train.shape[1]
        num_inducing = 64
        ind_process_dim = 17
        model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)
        model_name = f"/lmcgp_Normal_T{order}_M{num_inducing}_Q{ind_process_dim}_H{horizon}.pkl"
        # model = ind_gp(input_dim, observation_dim, num_inducing, X_train)
        # model_name = f"/indgp_Normal_T{order}_M{num_inducing}_H{horizon}.pkl"
        # model_name = f"/igp_T{order}_M{num_inducing}_H{horizon}.pkl" 
       
        dump_load_model(path, model_name, model, train_data)
        # plot_lengthscales(
                # model=model,
                # ind_process_dim=ind_process_dim,
                # path=path,
                # filename=f"/lengthscale_matrix_{horizon}",
                # )
        # plt.figure(figsize=(16, 8))
        # sns.heatmap(
                # np.abs(model.kernel.W.numpy()),
                # cbar=True,
                # cmap="viridis",
                # vmin=0,
                # vmax=1.5
                # )
        # tikz.save(path + f"/coregionalization_{horizon}.tex")
        # plt.savefig(path + f"/coregionalization_{horizon}.png")
        # plt.close()

        nlpd = negative_log_predictive_density(model, X_test, Y_test)
        msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
        crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
        mse = mean_squared_error(model, X_test, Y_test)
        mae = mean_absolute_error(model, X_test, Y_test)

        results_df.loc[horizon, "NLPD"] = nlpd.numpy()
        results_df.loc[horizon, "MSLL"] = msll
        results_df.loc[horizon, "CRPS"] = crps
        results_df.loc[horizon, "MSE"] = mse.numpy()
        results_df.loc[horizon, "MAE"] = mae.numpy()

       # Save results to Excel file
        try:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='a', if_sheet_exists="replace") as writer:
                results_df.to_excel(writer)
        except FileNotFoundError:
            with pd.ExcelWriter(path + filename + ".xlsx",
                            mode='w') as writer:
                results_df.to_excel(writer)


def plot_lmc_performance_bars():
    file_path = "./lmc_tests/lmc_vs_idp.xlsx"
    file_save = "./lmc_tests/performance_bars"
    xl = pd.ExcelFile(file_path)

    # Load each sheet into a DataFrame
    nlpd_df = xl.parse('NLPD')
    msll_df = xl.parse('MSLL')

    # Function to plot data
    def plot_data(df, metric_name):
        # Set the position of the bars on the x-axis
        barWidth = 0.25
        r1 = range(len(df))
        r2 = [x + barWidth for x in r1]
        r3 = [x + barWidth for x in r2]
    
        # Create the bar plot
        plt.figure(figsize=(16, 8))
        plt.bar(r1, df['IGP'], color='b', width=barWidth, edgecolor='grey', label='IGP')
        plt.bar(r2, df['IGP+'], color='g', width=barWidth, edgecolor='grey', label='IGP+')
        plt.bar(r3, df['LMCGP'], color='r', width=barWidth, edgecolor='grey', label='LMCGP')
    
        # Add labels
        plt.xlabel('H', fontweight='bold')
        plt.xticks([r + barWidth for r in range(len(df))], df['H'])
        plt.ylabel(metric_name, fontweight='bold')
        plt.title(f'{metric_name} for different H values')
        plt.savefig(file_save + f"_{metric_name}.png")
        tikz.save(file_save + f"_{metric_name}.tex")
        plt.close()

    # Plot NLPD data
    plot_data(nlpd_df, 'NLPD')
    # Plot MSLL data
    plot_data(msll_df, 'MSLL')


def plot_forecasting():
    path = './lmc_tests/'
    ind_process_dim = 17
    model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)
    model_name = f"/lmcgp_Normal_T{order}_M{num_inducing}_Q{ind_process_dim}_H1.pkl"
    dump_load_model(path + "across_horizons", model_name, model, train_data) 
    y_mean, y_var = model.predict_y(X_test)

    for task in range(observation_dim):
        output_name = norm_data.columns[task]
        y_lower = y_mean[:, task] - 1.96 * np.sqrt(y_var[:, task])
        y_upper = y_mean[:, task] + 1.96 * np.sqrt(y_var[:, task])
        time_range = range(len(X_test))
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.plot(time_range, Y_test[:, task], 'r.-')
        ax.plot(time_range, y_mean[:, task], 'b.-')
        ax.fill_between(
                time_range,
                y_lower,
                y_upper,
                alpha=0.5,
                color='b'
                )

        ax.set_title(
                f'Task {output_name}'
                )
        ax.set_xmargin(0.01),
        plt.savefig(path + f"lmc_forecasting_{output_name}.png")
        tikz.save(path + f"lmc_forecasting_{output_name}.tex")
        plt.close()




def main():

    # Set seed for NumPy
    np.random.seed(1966)
    # Set seed for GPflow
    tf.random.set_seed(1966)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    ct_data = load_datasets()
    indexes = [2, 9, 11, 12, 13, 14, 16, 17, 19, 21, 3, 6, 0, 5, 7, 8, 10, 1, 4, 18, 22, 20, 15]
    ct_data = ct_data.iloc[:, indexes]
    
    norm_data, data_mean, data_std = normalize_dataset(ct_data)
    data_mean = data_mean.values
    data_std = data_std.values
    
    # max_data = ct_data.max()
    # norm_data = ct_data / max_data
    # eps = 1e-3
    # norm_data.fillna(eps, inplace=True)
    # norm_data[norm_data <= 0] = eps

    # Create windows
    order = 1
    input_width = order
    label_width = 1
    shift = 1
    window = WindowGenerator(
        input_width,
        label_width,
        shift,
        norm_data.columns
    )
    print(window)
    x, y = window.make_dataset(norm_data)
    N = len(x)
    X_train, Y_train = x[0:int(N * 0.9)], y[0:int(N * 0.9)]
    X_test, Y_test = x[int(N * 0.9):], y[int(N * 0.9):]
    train_data = (X_test, Y_test)

    _, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]
    num_inducing = 64

    # lmc_model()
    plot_forecasting()

if __name__ == "__main__":
    # plot_lmc_performance_bars()
    # train_lmc_by_horizon()
    # main()
    # Set seed for NumPy
    np.random.seed(1966)
    # Set seed for GPflow
    tf.random.set_seed(1966)
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

    ct_data = load_datasets()
    indexes = [2, 9, 11, 12, 13, 14, 16, 17, 19, 21, 3, 6, 0, 5, 7, 8, 10, 1, 4, 18, 22, 20, 15]
    ct_data = ct_data.iloc[:, indexes]
    
    norm_data, data_mean, data_std = normalize_dataset(ct_data)
    data_mean = data_mean.values
    data_std = data_std.values
    
    # max_data = ct_data.max()
    # norm_data = ct_data / max_data
    # eps = 1e-3
    # norm_data.fillna(eps, inplace=True)
    # norm_data[norm_data <= 0] = eps

    # Create windows
    order = 1
    input_width = order
    label_width = 1
    shift = 1
    window = WindowGenerator(
        input_width,
        label_width,
        shift,
        norm_data.columns
    )
    print(window)
    x, y = window.make_dataset(norm_data)
    N = len(x)
    X_train, Y_train = x[0:int(N * 0.9)], y[0:int(N * 0.9)]
    X_test, Y_test = x[int(N * 0.9):], y[int(N * 0.9):]
    train_data = (X_test, Y_test)

    _, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]
    num_inducing = 64

    # lmc_model()
    plot_forecasting()

