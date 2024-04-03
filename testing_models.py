import numpy as np
import matplotlib.pyplot as plt
import gpflow as gpf
from likelihoods import LogNormalLikelihood
from data_exploration import get_uv_data
from metrics import *

# Get train and test datasets
train_data, val_data, test_data = get_uv_data()
X_train, Y_train = test_data
X_test, Y_test = test_data

n_samples, input_dim = X_test.shape
observation_dim = Y_test.shape[1]
num_inducing = 50
indfun_dim = 25

observation_dim, input_dim, num_inducing, indfun_dim
# Training and Testing the Models
for model, name in zip([model_ind, model_lmc, model_hc_ind, model_hc_cor], ["Independent", "LMC", "Chained_Ind", "Chainde_Corr"]):
    train_model(model, train_data, epochs=500, verbose=False)
    nlogpred = negatve_log_predictive_density(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)
    print(f"{name} - NLPD: {nlogpred.numpy():.2e}, MSE: {mse.numpy():.2e}, MAE: {mae.numpy():.2e}")
    plot_gp_predictions(model, X_test, Y_test, name)
    # plot_results(model, X_test, Y_test)