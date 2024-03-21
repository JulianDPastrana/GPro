import tensorflow as tf

def negatve_log_predictive_density(model, X_test, Y_test, n_samples=100):
        F_samples = model.predict_f_samples(X_test, n_samples)
        # monte-carlazo
        log_pred = model.likelihood.log_prob(X=X_test, F=F_samples, Y=Y_test)
        nlogpred = -tf.reduce_sum(log_pred) / n_samples
        return nlogpred