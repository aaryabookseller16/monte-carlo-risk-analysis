import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pathlib import Path

PLOTS_DIR = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

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
    
    plt.savefig(PLOTS_DIR / "return_distribution.pdf")
    plt.close()
    
    
def plot_mc_fan(
    paths: np.ndarray,
    n_show: int = 100,
) -> None:
    """
    Plot a fan chart of Monte Carlo price paths.

    Parameters
    ----------
    paths : np.ndarray
        Array of shape (T+1, N) with simulated price paths.
    n_show : int
        Number of paths to display (subset for readability).
    """
    T, N = paths.shape
    n_show = min(n_show, N)

    plt.figure(figsize=(9, 5))

    # Plot a subset of paths
    for i in range(n_show):
        plt.plot(paths[:, i], color="steelblue", alpha=0.15, linewidth=1)

    plt.xlabel("Time (days)")
    plt.ylabel("Price")
    plt.title("Monte Carlo Price Paths (Fan Chart)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "mc_fan_chart.pdf")
    plt.close()


def plot_terminal_distribution(paths: np.ndarray) -> None:
    """
    Plot the distribution of terminal prices from Monte Carlo paths.
    """
    terminal = paths[-1, :]

    plt.figure(figsize=(8, 5))
    plt.hist(terminal, bins=60, density=True, alpha=0.7, color="steelblue")

    plt.xlabel("Terminal Price")
    plt.ylabel("Density")
    plt.title("Distribution of Terminal Prices (Monte Carlo)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "terminal_distribution.pdf")
    plt.close()

if __name__ == "__main__":
    from src.data import load_prices
    from src.returns import log_returns, estimate_gaussian_params
    from src.simulate import simulate_price_paths

    # Load data
    prices = load_prices("SPY", start="2015-01-01")
    returns = log_returns(prices)
    mu, sigma = estimate_gaussian_params(returns)

    s0 = float(prices["adj_close"].iloc[-1])

    # Simulate paths
    paths = simulate_price_paths(
        s0=s0,
        mu=mu,
        sigma=sigma,
        n_days=252,
        n_paths=10_000,
        seed=42,
    )

    # Plots
    plot_return_distribution(returns, mu, sigma)
    plot_mc_fan(paths, n_show=100)
    plot_terminal_distribution(paths)