import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_return_distribution(returns: pd.Series, mu:float, sigma: float) -> None:
    returns = returns.dropna()
    
    if returns.empty:
        raise ValueError("Returns series is empty")
    
    plt.figure(figsize=(8,5))
    
    plt.hist(
        returns,
        bins=60,
        density=True,
        alpha=0.6,
        color="steelblue",
        label="Empirical Returns"
    )
    
    x = np.linspace(
        returns.min(),
        returns.max(),
        500
    )
    
    pdf = norm.pdf(x, loc=mu, scale=sigma)
    
    plt.plot(
        x,
        pdf,
        color="darkred",
        linewidth=2,
        label="Gaussian (μ, σ)",
    )
    
    plt.xlabel("Daily log return")
    plt.ylabel("Density")
    plt.title("Empirical Return Distribution vs Gaussian Assumption")
    plt.legend()
    plt.tight_layout()
    
    plt.show()
    plt.savefig("plots/return_distribution.png")
    
    
if __name__ == "__main__":
    from src.data import load_prices
    from src.returns import log_prices, estimate_gaussian_params

    # 1. Load price data
    prices = load_prices("SPY", start="2015-01-01")

    # 2. Compute log returns
    returns = log_prices(prices)

    # 3. Estimate Gaussian parameters
    mu, sigma = estimate_gaussian_params(returns)

    # 4. Plot empirical vs Gaussian distribution
    plot_return_distribution(returns, mu, sigma)