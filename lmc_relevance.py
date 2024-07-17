import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
import pandas as pd
from likelihoods import MOChainedLikelihoodMC
from data_exploration import get_daily_vol_data
from sklearn.model_selection import TimeSeriesSplit
from itertools import product
import string
import tensorflow_probability as tfp
from metrics import *
from main import *

tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
path = "./results/lmc_tests"


def main():

    # Set seed for NumPy
    np.random.seed(0)

    # Set seed for GPflow
    tf.random.set_seed(0)
    # Load the LMC model

    ct_data = load_datasets()
    indexes = [2, 9, 11, 12, 13, 14, 16, 17, 19, 21, 3, 6, 0, 5, 7, 8, 10, 1, 4, 18, 22, 20, 15]
    ct_data = ct_data.iloc[:, indexes]
    norm_data, data_mean, data_std = normalize_dataset(ct_data)
    data_mean = data_mean.values
    data_std = data_std.values
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
    ind_process_dim = observation_dim

    model_w = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)

    model_name = f"/lmcgp_Normal_T{order}_Q{ind_process_dim}_M{num_inducing}.pkl"
    with open(path + model_name, 'rb') as file:
        params = pickle.load(file)
    gpf.utilities.multiple_assign(model_w, params)
    
    W = model_w.kernel.W.numpy()
    rq = np.sum(W**2, axis=0)

    sort_idx = np.argsort(rq)[::-1]

    labels =  list(string.ascii_uppercase[:observation_dim])
    numbers = np.arange(observation_dim)
    plt.bar(numbers, rq[sort_idx])
    plt.xticks(ticks=numbers, labels=numbers[sort_idx])
    plt.savefig(path + "/relevance.png")

    W_sort = W[:, sort_idx]
    X_train_sort = X_train[:, sort_idx]
    X_test_sort = X_test[:, sort_idx]
    results_df = pd.DataFrame()
    filename = "/lmc_relevance"
    for q in np.arange(1, observation_dim+1)[::-1]:
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
        gpf.set_trainable(model.kernel.W, False)
         
        try:
            with open(path + model_name, 'rb') as file:
                params_ = pickle.load(file)
            gpf.utilities.multiple_assign(model, params_)
            print("Model parameters loaded successfully.")
        
        except FileNotFoundError:    
        
            train_model(model, (X_trainq, Y_train), batch_size=64, epochs=2500, patience=50, lr=0.05)
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

if __name__ == "__main__": 
    main()      