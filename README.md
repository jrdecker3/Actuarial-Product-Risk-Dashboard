## Enterprise Risk & Reserving Dashboard

An end-to-end actuarial workstation built in Python and Streamlit, designed to model historical claims development and forecast stochastic risk. This platform processes NAIC Schedule P liability data through Stochastic Loss Modeling, and historical claims data providing interactive loss development triangles and Monte Carlo simulations.

## Executive Summary

Traditional reserving models often fail to capture Catastrophic Risk in volatile insurance lines. This dashboard bridges the gap between pure econometrics and enterprise risk strategy by synthesizing **35 distinct statistical distributions** (e.g., Weibull, Gamma, Log-Normal) to compete via Goodness-of-Fit testing (AIC). This Model depicts the historical data of 70 insurance firms, to predict capital requirements neccessary for setting reserves.

## Core Features

### 1. Actuarial Loss Development Triangles
-Aggregates incremental paid losses across multiple accident years and development lags.
-Generates interactive, heat-mapped pivot tables to identify adverse development and settlement velocity(Darker Blue indicates larger losses)
-Plots historical claim maturation curves by Accident Year.

### 2. Stochastic Monte Carlo Engine (Severity & Frequency)
Runs maximum-likelihood calculations to fit and rank 35 actuarial distributions, automatically selecting the optimal shape for a specific company's risk profile.
* **Frequency/Severity Simulation:** Executes Monte Carlo stress tests utilizing the optimal severity curve and a Poisson frequency parameter.

* **Value at Risk (VaR) Analysis:** Calculates 95%, 99%, and 99.9% percentiles to determine capital reserve requirements for catastrophic event.

## Technology Stack
* **Python** (Data processing and calculus engine)
* **Streamlit** (Interactive frontend web framework)
* **Pandas & NumPy** (data manipulation and matrix operations)
* **SciPy** (Statistical distribution fitting and AIC testing)
* **Matplotlib** (Data visualization and probability density plotting)

## 🚀 How to Run Locally

1. Clone this repository to your local machine:
   ```bash
   git clone [https://github.com/jrdecker3/Actuarial-Risk-Dashboard.git](https://github.com/jrdecker3/Actuarial-Risk-Dashboard.git)