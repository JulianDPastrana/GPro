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
