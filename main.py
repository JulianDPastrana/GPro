import numpy as np
import tensorflow as tf
import gpflow as gpf
import tensorflow_probability as tfp
import pickle
import seaborn as sns
from data_exploration import get_daily_vol_data
from metrics import *
# from likelihoods import MOChainedLikelihoodMC
from grid_search import *

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
path = "./results"


def main():
    # Set seed for NumPy
    np.random.seed(0)

    # Set seed for GPflow
    tf.random.set_seed(0)

    order = 2
    train_data, test_data = get_daily_vol_data(input_width=order, label_width=1, shift=1)
    X_test, Y_test = test_data
    X_train, Y_train = train_data

    _, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]
    num_inducing = 2**3
    ind_process_dim = 2**8

    latent_dim = 2 * observation_dim

    # model = chained_corr(
    #     input_dim,
    #     latent_dim,
    #     observation_dim,
    #     ind_process_dim,
    #     num_inducing,
    #     X_train
    # )

    model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)

    # model = ind_gp(input_dim, observation_dim, num_inducing, X_train)

    # model_name = f"/chdgp_Normal_T{order}_Q{latent_dim}_M{num_inducing}.pkl"
    # model_name = f"/indgp_Normal_T{order}_M{num_inducing}.pkl"
    model_name = f"/lmcgp_Normal_T{order}_Q{ind_process_dim}_M{num_inducing}.pkl"
    # with open(path + f"/indgp_Normal_T{order}_M{num_inducing}.pkl", 'rb') as file:
    #         params = pickle.load(file)
    # gpf.utilities.multiple_assign(model, params)
    # try:
    #     with open(path + model_name, 'rb') as file:
    #         params = pickle.load(file)
    #     gpf.utilities.multiple_assign(model, params)
    #     print("Model parameters loaded successfully.")
    
    # except FileNotFoundError:    
    #     train_model(model, train_data, batch_size=64, epochs=150)
    #     with open(path + model_name, 'wb') as handle:
    #         pickle.dump(gpf.utilities.parameter_dict(model), handle)
    #     print("Model trained and parameters saved successfully.")

    gpf.set_trainable(model.kernel.W, False)
    train_model(model, train_data, batch_size=64, epochs=150, patience=20)
    gpf.set_trainable(model, False)
    gpf.set_trainable(model.kernel.W, True)
    train_model(model, train_data, batch_size=64, epochs=150, patience=20)
    sns.heatmap(model.kernel.W.numpy(), annot=True, fmt=".2f")
    plt.show()
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
            Y_mean[:, task],
            Y_var[:, task],
            task_name=str(task),
            y_true=Y_test[:, task],
            fname=path+str(task)+".png"
        )

if __name__ == "__main__": 
    main()