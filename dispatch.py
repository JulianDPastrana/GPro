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
    params = load_thermo_params()
    print(params)

if __name__ == "__main__": 
    main()