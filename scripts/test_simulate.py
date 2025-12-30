from __future__ import annotations

import numpy as np

from src.data import load_prices
from src.returns import log_returns, estimate_gaussian_params
from src.simulate import simulate_price_paths


def main() -> None:
    # 1) Load prices (cached) and pick starting price
    prices = load_prices("SPY", start="2015-01-01")
    s0 = float(prices["adj_close"].iloc[-1])

    # 2) Compute log returns and estimate daily Gaussian params
    r = log_returns(prices)
    mu, sigma = estimate_gaussian_params(r)

    # 3) Simulate Monte Carlo paths
    n_days = 252
    n_paths = 10_000
    paths = simulate_price_paths(
        s0=s0,
        mu=mu,
        sigma=sigma,
        n_days=n_days,
        n_paths=n_paths,
        seed=42,
    )

    # 4) Hard checks (fail fast if anything is off)
    assert paths.shape == (n_days + 1, n_paths), f"Unexpected shape: {paths.shape}"
    assert np.allclose(paths[0, :], s0), "Row 0 must equal s0 for all paths"
    assert not np.isnan(paths).any(), "paths contains NaNs"
    assert not np.isinf(paths).any(), "paths contains infs"

    # Day 1 should vary (randomness starts)
    assert np.std(paths[1, :]) > 0.0, "Day 1 prices should vary across paths"

    terminal = paths[-1, :]
    assert np.std(terminal) > 0.0, "Terminal prices should vary across paths"
    assert np.min(terminal) > 0.0, "Prices must stay positive under exp model"

    # 5) Human-readable sanity output
    print("OK: Monte Carlo simulation sanity checks passed.")
    print(f"s0={s0:.4f}  mu={mu:.6g}  sigma={sigma:.6g}")
    print(f"paths.shape={paths.shape}")
    print("paths[:3, :3] =\n", paths[:3, :3])
    print(
        "terminal stats:",
        f"mean={terminal.mean():.4f}",
        f"std={terminal.std():.4f}",
        f"p05={np.percentile(terminal, 5):.4f}",
        f"p50={np.percentile(terminal, 50):.4f}",
        f"p95={np.percentile(terminal, 95):.4f}",
    )


if __name__ == "__main__":
    main()