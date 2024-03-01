import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
from data_exploration import get_uv_data

# Generate random data for demonstration
rng = np.random.default_rng()
# # Build inputs X
# N = 1000
# X = np.linspace(0, 4 * np.pi, N)
# # Deterministic functions in place of latent ones
# f1 = np.sin
# f2 = np.cos

# # Use transform = exp to ensure positive-only scale values
# transform = np.exp

# # Compute loc and scale as functions of input X
# loc = f1(X)
# scale = transform(f2(X))

# # Sample outputs Y from Gaussian Likelihood
# Y = np.random.lognormal(loc, scale)
train_data, _ = get_uv_data()
_, Y = train_data
# List of distributions to test against
distributions = [stats.lognorm, stats.burr, stats.exponweib, stats.invweibull]

# Initialize list to store results
results = []

# Loop through each distribution and perform goodness of fit test
for distribution in distributions:
    # Perform goodness of fit test
    res = stats.goodness_of_fit(distribution, Y[:, 0], random_state=rng)
    print(res.fit_result)
    # res.fit_result.plot()
    # plt.show()
    # Extract relevant results
    name = distribution.name
    statistic = res.statistic
    pvalue = res.pvalue
    
    # Append results
    results.append((name, statistic, pvalue))

# Convert results to DataFrame for display
results_df = pd.DataFrame(results, columns=['Distribution', 'Statistic', 'p-Value'])

# Display the DataFrame
print(results_df)
