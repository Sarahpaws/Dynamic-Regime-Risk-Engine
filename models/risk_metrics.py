import numpy as np
import pandas as pd

def calculate_volatility(returns):
    return returns.std() * np.sqrt(252)


def calculate_covariance(returns):
    return returns.cov() * 252


def calculate_correlation(returns):
    return returns.corr()


def rolling_volatility(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)


def portfolio_volatility(weights, cov_matrix):
    weights = np.array(weights)
    return np.sqrt(weights.T @ cov_matrix @ weights)