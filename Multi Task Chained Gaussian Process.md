# Dataset Description
Consider a dataset with observation dimension $D$, defined as: $$ S = \{(\mathbf{x}_{d, n}, y_{d, n})\}_{n=1, d=1}^{N_d, D} $$where each feature vector $\mathbf{x}_d$ belongs to the input space $\mathcal{X}_d$, and each corresponding output $y_d$ belongs to the output space $\mathcal{Y}_d$.

# Likelihood Model

Define the full input set as $\mathbf{x} = \{\mathbf{x}_d\}_{d=1}^D$ and a vector-valued function $\mathbf{y}(\mathbf{x}) = [y_1(\mathbf{x}_1), y_2(\mathbf{x}_2), \dots, y_D(\mathbf{x}_D)]^\top$. We assume the distribution over $y_d(\mathbf{x}_d)$ is specified by a set of parameters $\mathbf{\theta}_d(\mathbf{x}_d) \in \mathcal{\Theta}^{J_d}$, where $J_d$ denotes the number of parameters defining the distribution. Similarly, $\mathbf{\theta}(\mathbf{x}) = \{\mathbf{\theta}_d(\mathbf{x}_d)\}_{d=1}^D$.

The likelihood of observing $\mathbf{y}(\mathbf{x})$ given $\mathbf{\theta}(\mathbf{x})$ can be expressed as a product of individual likelihoods:
$$
p(\mathbf{y}(\mathbf{x}) \mid \mathbf{\theta}(\mathbf{x})) = \prod_{d=1}^D p(y_{d}(\mathbf{x}_d)\mid \mathbf{\theta}_{d}(\mathbf{x}_d)) = \prod_{d=1}^D \prod_{n=1}^{N_d}p(y_{d,n}\mid \mathbf{\theta}_{d}(\mathbf{x}_{d,n}))
$$
Each parameter $\theta_{d,j}(\mathbf{x}_d) \in \mathbf{\theta}_d(\mathbf{x}_d)$ is a deterministic transformation of a Gaussian Process (GP) prior realization $f_{d,j}(\mathbf{x}_d)$, given by $\theta_{d,j}(\mathbf{x}_d) = g_{d,j}(f_{d,j}(\mathbf{x}_d))$. Let be $\hat{\mathbf{f}}_d(\mathbf{x}_d) = [f_{d,1}(\mathbf{x}_d), f_{d,2}(\mathbf{x}_d), \dots, f_{d,J_d}(\mathbf{x}_d)]^\top \in \mathbb{R}^{J_d \times 1}$ and $\mathbf{f}(\mathbf{x}) = [f_{1,1}(\mathbf{x}_1), f_{1,2}(\mathbf{x}_1), \dots, f_{D,J_D}(\mathbf{x}_D)] \in \mathbb{R}^{J \times 1}$, where $J = \sum_{d=1}^D J_d$. The conditionally independent likelihood is then formulated as:
$$
p(\mathbf{y}(\mathbf{x}) \mid \mathbf{\theta}(\mathbf{x})) = p(\mathbf{y}(\mathbf{x}) \mid \mathbf{f}(\mathbf{x})) \prod_{d=1}^D p(y_{d}(\mathbf{x}_d)\mid \hat{\mathbf{f}}_{d}(\mathbf{x}_d)) = \prod_{d=1}^D \prod_{n=1}^{N_d}p(y_{d,n}\mid \hat{\mathbf{f}}_{d}(\mathbf{x}_{d,n}))
$$
This formulation introduces $J$ latent parameter functions $f_{d,j}(\mathbf{x}_d)$, each governed by a GP prior.

# Multi-Latent Parameter GP Prior

Each latent function $f_{d,j}(\mathbf{x}_d)$ is assumed to be drawn form an zero mean independent  GP prior for simplicity such that:
$$
f_{d,j}(\cdot) \sim \mathcal{GP}(0, k_{d,j}(\cdot, \cdot))
$$
where $k_{d,j}$ can be any valid covariance function.

## Data Realizations

For further detail, let's define:

- $\mathbf{f}_{d,j} = [f_{d,j}(\mathbf{x}_{d, 1}), f_{d,j}(\mathbf{x}_{d,2}), \dots, f_{d,j}(\mathbf{x}_{d, N_d})]^\top$ as the vector of function evaluations for the $j$-th function in the $d$-th dataset, residing in $\mathbb{R}^{N_d}$.

- $\hat{\mathbf{f}}_{d} = [\mathbf{f}_{d,1}^\top, \mathbf{f}_{d,2}^\top, \dots, \mathbf{f}_{d,J_d}^\top]^\top$ as the stacked vector of all function evaluations in the $d$-th dataset, belonging to $\mathbb{R}^{J_dN_d}$.

- $\mathbf{f} = [\hat{\mathbf{f}}_1^\top, \hat{\mathbf{f}}_2^\top, \cdots, \hat{\mathbf{f}}_D^\top]^\top$ as the grand vector of all evaluations across all datasets, positioned in $\mathbb{R}^{L}$, where $L = \sum_{d=1}^D J_dN_d$ encapsulates the total dimensionality.

## Generative Model

The underlying generative model is described as follows: we draw $\mathbf{f}$ from a multivariate normal distribution, $\mathbf{f} \sim \mathcal{N}(\mathbf{0}, \mathbf{K})$, with $\mathbf{K}$ being a block-diagonal covariance matrix. This matrix $\mathbf{K}$ integrates $D$ sub-matrices on its diagonal, each corresponding to a dataset. Each of these sub-matrices is itself block-diagonal, containing $J_d$ matrices along its diagonal corresponding to a latent parameter function of task $d$, each of size $N_d$, to account for the covariance structure among the $J_d$ functions within the $d$-th dataset.

The covariance matrix $\mathbf{K}$ is structured as follows:

$$
\mathbf{K} = 
\left[
\begin{array}{cccc}
    \begin{bmatrix}
        K_{1,1} & 0 & \cdots & 0 \\
        0 & K_{1,2} & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & K_{1,J_1} \\
    \end{bmatrix} & \mathbf{0} & \cdots & \mathbf{0} \\
    \mathbf{0} & \begin{bmatrix}
        K_{2,1} & 0 & \cdots & 0 \\
        0 & K_{2,2} & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & K_{2,J_2} \\
    \end{bmatrix} & \cdots & \mathbf{0} \\
    \vdots & \vdots & \ddots & \vdots \\
    \mathbf{0} & \mathbf{0} & \cdots & \begin{bmatrix}
        K_{D,1} & 0 & \cdots & 0 \\
        0 & K_{D,2} & \cdots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0 & \cdots & K_{D,J_D} \\
    \end{bmatrix} \\
\end{array}
\right]
$$

Here, each block matrix $K_{d,j}$ is an $N_d \times N_d$ covariance matrix, representing the covariance matrix for the $j$-th latent parameter function within the $d$-th dataset. The matrices $\mathbf{0}$ represent blocks of zeros, indicating the conditional independence between different datasets and their latent functions.

$$
\mathbf{K} = \sum_{d=1}^{D} \text{Diag}(\mathbf{e}_{d, D}) \bigotimes \sum_{j=1}^{J_d} \text{Diag}(\mathbf{e}_{j, J_d}) \bigotimes\mathbf{K}_{d,j}
$$
Here $\mathbf{e}_{d,D}$ is the $d$-th canonical basis of $\mathbb{R}^D$ and $\text{Diag}(\cdot)$ operator transforms a vector into a diagonal matrix. The above analysis applies to $\text{Diag}(\mathbf{e}_{j, J_d})$.

One we obtain the sample for $\mathbf{f}$, we evaluate the vector or parameters $\mathbf{\theta} = [\hat{\mathbf{\theta}}_1^\top, \hat{\mathbf{\theta}}_2^\top, \cdots, \hat{\mathbf{\theta}}_D^\top]^\top \in \mathbb{R}^{L}$, where $\hat{\mathbf{\theta}}_{d} = [\mathbf{\theta}_{d,1}^\top, \mathbf{\theta}_{d,2}^\top, \dots, \mathbf{\theta}_{d,J_d}^\top]^\top \in \mathbb{R}^{J_d N_d}$, and $\mathbf{\theta}_{d,j} = [\theta_{d,j}(\mathbf{x}_{d, 1}), \theta_{d,j}(\mathbf{x}_{d,2}), \dots, \theta_{d,j}(\mathbf{x}_{d, N_d})]^\top \in  \mathbb{R}^{N_d}$. Having specified $\mathbf{\theta}$, we can generate samples for the output vector $\mathbf{y} = [y_1, y_2, \cdots, y_D]^\top \in \prod_{d=1}^D \mathcal{Y}_d$, where elements in $y_d$ are obtained by sampling from the conditional distributions $p(y_{d}(\mathbf{x}_d)\mid \mathbf{\theta}_{d}(\mathbf{x}_d))$.
# Scalable Variational Inference


