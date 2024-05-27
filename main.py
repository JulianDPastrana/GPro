import numpy as np
import tensorflow as tf
import gpflow as gpf
import tensorflow_probability as tfp
import pickle
from data_exploration import get_daily_vol_data
from metrics import negative_log_predictive_density, mean_squared_error, mean_absolute_error
from metrics import train_model
from likelihoods import MOChainedLikelihoodMC

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

    n_samples, input_dim = X_train.shape
    observation_dim = Y_train.shape[1]
    num_inducing = 1024
    ind_process_dim = 128

    latent_dim = 2 * observation_dim

    kern_list = [gpf.kernels.SquaredExponential(lengthscales=np.ones(input_dim)) + 
                 gpf.kernels.Linear(variance=np.ones(input_dim)) + 
                 gpf.kernels.Constant() for _ in range(ind_process_dim)]
    
    kernel = gpf.kernels.LinearCoregionalization(
        kern_list, W=np.random.randn(latent_dim, ind_process_dim) + np.eye(latent_dim, ind_process_dim)
    )
    
    Zinit = np.random.rand(num_inducing, input_dim)
    Zs = [Zinit.copy() for _ in range(ind_process_dim)]
    iv_list = [gpf.inducing_variables.InducingPoints(Z) for Z in Zs]
    iv = gpf.inducing_variables.SeparateIndependentInducingVariables(iv_list)
    
    q_mu = np.zeros((num_inducing, ind_process_dim))
    q_sqrt = np.repeat(np.eye(num_inducing)[None, ...], ind_process_dim, axis=0) * 1.0
    
    likelihood = likelihood = MOChainedLikelihoodMC(
            input_dim=input_dim,
            latent_dim=latent_dim,
            observation_dim=observation_dim,
            distribution_class=tfp.distributions.Beta,
            param1_transform=tfp.bijectors.Softplus(),
            param2_transform=tfp.bijectors.Softplus()
        )
    
    model = gpf.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=iv,
        q_mu=q_mu,
        q_sqrt=q_sqrt
    )

    train_model(model, train_data, batch_size=64, epochs=500)

    nlpd = negative_log_predictive_density(model, X_test, Y_test)
    mse = mean_squared_error(model, X_test, Y_test)
    mae = mean_absolute_error(model, X_test, Y_test)

    print(f"NLPD: {nlpd}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")


    with open(path + f"/chd_beta_Q{latent_dim}_M{num_inducing}.pkl", 'wb') as handle:
        pickle.dump(gpf.utilities.parameter_dict(model), handle)

if __name__ == "__main__": 
    main()
