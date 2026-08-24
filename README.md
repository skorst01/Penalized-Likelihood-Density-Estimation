# Penalized-Likelihood-Density-Estimation
This repository represents supplement material for paper Penalized likelihood estimation of probability density functions using compositional splines by Škorňa S. and Machalová J. (2026).

Please note that the repository contains MATLAB codes for reproducing the main simulation and application results presented in the paper. The code is provided for reproducibility purposes and is not intended as a general-purpose software package.

## Repository structure

The repository contains the following files:

- `simulation_univariate.m`  
  Reproduces the univariate simulation study, including the selection of the penalization parameter by K-fold cross-validation and the evaluation of the resulting estimates.

- `simulation_bivariate.m`  
  Reproduces the bivariate simulation study, including the selection of the penalization parameters by K-fold cross-validation and the evaluation of the resulting estimates.

- `application_univariate.m`  
  Reproduces the univariate application to the geochemical data.

- `application_bivariate.m`  
  Reproduces the bivariate application to the geochemical data.

- `ML_1D_grad.m`  
  Evaluates the univariate penalized negative log-likelihood function and its gradient.

- `ML_pen_fun_grad.m`  
  Evaluates the bivariate penalized negative log-likelihood function and its gradient.

- `trapz_weights_1D.m`  
  Computes the trapezoidal integration weights used in the univariate implementation.

- `trapz_weights.m`  
  Computes the trapezoidal integration weights used in the bivariate implementation.

- `data.csv`  
  Contains the geochemical data used in the applications.


## Requirements

The code was implemented in MATLAB and requires the following toolboxes:

- Optimization Toolbox
- Statistics and Machine Learning Toolbox
- Curve Fitting Toolbox


## Usage

Each simulation or application can be reproduced by running the corresponding MATLAB script:

- `simulation_univariate.m` for the univariate simulation study,
- `simulation_bivariate.m` for the bivariate simulation study,
- `application_univariate.m` for the univariate application,
- `application_bivariate.m` for the bivariate application.

The main simulation settings, including the sample size, number of cross-validation folds, spline degree, difference order, and considered numbers of inner knots, can be specified at the beginning of the corresponding script.


## Data

The file `data.csv` contains the geochemical data used in the empirical applications. The univariate application considers lead (Pb), while the bivariate application considers the joint distribution of lead (Pb) and zinc (Zn) concentrations.

