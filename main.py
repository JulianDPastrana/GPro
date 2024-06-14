import numpy as np
import tensorflow as tf
import gpflow as gpf
import tensorflow_probability as tfp
import pickle
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

    train_data, test_data = get_daily_vol_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data

    _, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]
    num_inducing = 150
    ind_process_dim = 32

    latent_dim = 2 * observation_dim

    model = chained_corr(
        input_dim,
        latent_dim,
        observation_dim,
        ind_process_dim,
        num_inducing,
        X_train
    )

    # model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)

    # model = ind_gp(input_dim, observation_dim, num_inducing, X_train)

    model_name = f"/chd_LogitNormal_Q{latent_dim}_M{num_inducing}.pkl"

    try:
        with open(path + model_name, 'rb') as file:
            params = pickle.load(file)
        gpf.utilities.multiple_assign(model, params)
        print("Model parameters loaded successfully.")
    
    except FileNotFoundError:    
        train_model(model, train_data, batch_size=8, epochs=150)
        # with open(path + model_name, 'wb') as handle:
        #     pickle.dump(gpf.utilities.parameter_dict(model), handle)
        print("Model trained and parameters saved successfully.")

    nlpd = negative_log_predictive_density(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)

    print(f"NLPD: {nlpd}")
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