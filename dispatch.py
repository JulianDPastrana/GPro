import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import gpflow as gpf
import seaborn as sns
from likelihoods import LogNormalLikelihood, LogNormalMCLikelihood, LogNormalQuadLikelihood, HeteroskedasticLikelihood
from data_exploration import get_uv_data
from gpflow.utilities import print_summary
import tensorflow_probability as tfp
import matplotlib
from main import build_model
import pickle
from matplotlib import cm
from scipy.stats import gamma
from metrics import negatve_log_predictive_density, train_model
import pandas as pd
from scipy.optimize import linprog


def load_thermo_params() -> pd.DataFrame:
    thermo_params = pd.read_excel(
        io='./Info parametros por térmica.xlsx',
        sheet_name='info termicas',
        header=0,
        index_col=0,
        usecols='A:C'
    ).drop(index='TERMOYOPAL G5')
    return thermo_params


def main():
    thermo_params = load_thermo_params()
    num_thermo = len(thermo_params)
    C = thermo_params['Heat Rate (MBTU/MWh)'].values
    Ub = thermo_params['Capacidad Efectiva Neta (MW)'].values * 1e3 * 24
    Lb = np.zeros((num_thermo,))
    Aeq = np.ones((1, num_thermo))

    save_dir = "saved_model_0"
    model = tf.saved_model.load(save_dir)
    train_data, val_data, test_data = get_uv_data()
    X_test, Y_test = test_data
    X_train, Y_train = train_data
    # Cargar el objeto desde un archivo
    with open('save_dir', 'rb') as file:
        params = pickle.load(file)
    model = build_model(train_data)
    gpf.utilities.multiple_assign(model, params)

    n_samples = 30
    Fsamples = model.predict_f_samples(X_test, n_samples)
    
    alpha = np.exp(Fsamples[..., 0].numpy().squeeze())
    beta = np.exp(Fsamples[..., 1].numpy().squeeze())
    dist = tfp.distributions.Gamma(alpha, beta)



if __name__ == "__main__": 
    main()