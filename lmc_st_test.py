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

def main():
    # Set seed for NumPy
    np.random.seed(0)

    # Set seed for GPflow
    tf.random.set_seed(0)

    path = "/home/usuario/Documents/GPro/results/lmc_single_test"
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

    results_df = pd.DataFrame()
    filename = "/lmc_Q_search"

    for ind_process_dim in range(1, observation_dim+1):

        model = lmc_gp(input_dim, observation_dim, ind_process_dim, num_inducing, X_train)

        model_name = f"/lmcgp_Normal_T{order}_Q{ind_process_dim}_M{num_inducing}.pkl"

        # train_model(model, train_data, batch_size=64, epochs=1500, patience=50, lr=0.01)

        # with open(path + model_name, 'wb') as handle:
        #     pickle.dump(gpf.utilities.parameter_dict(model), handle)
        # print("Model trained and parameters saved successfully.")

        with open(path + model_name, 'rb') as file:
            params = pickle.load(file)
        gpf.utilities.multiple_assign(model, params)
        print("Model parameters loaded successfully.")


        plt.figure(figsize=(16, 8))
        sns.heatmap(model.kernel.W.numpy(), annot=True, fmt=".2f")
        plt.savefig(path + f"/coregionalization_{ind_process_dim}.png")
        plt.close()

        nlpd = negative_log_predictive_density(model, X_test, Y_test)
        msll = mean_standardized_log_loss(model, X_test, Y_test, Y_train)
        crps = continuous_ranked_probability_score_gaussian(model, X_test, Y_test)
        mse = mean_squared_error(model, X_test, Y_test)
        mae = mean_absolute_error(model, X_test, Y_test)

        results_df.loc[ind_process_dim, "NLPD"] = nlpd.numpy()
        results_df.loc[ind_process_dim, "MSLL"] = msll
        results_df.loc[ind_process_dim, "CRPS"] = crps
        results_df.loc[ind_process_dim, "MSE"] = mse.numpy()
        results_df.loc[ind_process_dim, "MAE"] = mae.numpy()

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


