import numpy as np
import pandas as pd

def log_prices(prices: pd.DataFrame) -> pd.Series:
    if "adj_close" not in prices.columns:
        raise ValueError("Expected column 'adj_close'")
    
    s = prices['adj_close']
    r = np.log(s / s.shift(1)) # today's price divided by yesterday
    
    r = r.dropna()
    r.name = "log_return"
    
    return r

def estimate_gaussian_params(returns: pd.Series) -> tuple[float, float]:
    returns = returns.dropna()
    
    mu = float(returns.mean())
    sigma = float(returns.std(ddof=1))
    
    return mu, sigma

