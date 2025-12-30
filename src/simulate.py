import numpy as np

def simulate_price_paths(
    s0: float,
    mu: float,
    sigma: float,
    n_days: int = 252,
    n_paths: int = 10_000,
    seed: int = 42
) -> np.ndarray:
    
    rng = np.random.default_rng(seed)
    
    # iid assumption
    z = rng.standard_normal(size=(n_days, n_paths))
    
    # convert shocks into simulated log returns
    r = mu + sigma * z
    
    # turn log returns into log prices
    log_s0 = np.log(s0)
    log_paths = log_s0 + np.cumsum(r, axis = 0)
    
    # convert log back to prices
    paths = np.exp(log_paths)
    
    paths = np.vstack([np.full((1, n_paths), s0), paths])
    
    return paths
    