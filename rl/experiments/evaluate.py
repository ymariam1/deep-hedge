import pathlib
import sys
import os

# Add the project root to the Python path so we can import from src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

# Import HedgingEnv directly to avoid package issues
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from stable_baselines3 import PPO
from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
import numpy as np, pandas as pd, torch
from env.hedging_env import HedgingEnv

N_PATHS = 20_000
spot     = BrownianStock(sigma=0.2, mu=0.0, cost=1e-4)
deriv    = EuropeanOption(spot, maturity=1.0, strike=1.0)
env      = HedgingEnv(deriv, spot, steps=250, cost=1e-4, shortfall_thr=-0.10)

# Find the most recent model
results_dir = pathlib.Path("rl/results/ppo_models")
if results_dir.exists():
    model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    if model_dirs:
        # Get the most recent model directory
        latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
        model_path = latest_model_dir / "ppo_band_final.zip"
        if model_path.exists():
            print(f"Loading model from: {model_path}")
            model = PPO.load(str(model_path))
        else:
            print(f"Model file not found: {model_path}")
            print("Available files in directory:")
            for f in latest_model_dir.iterdir():
                print(f"  {f.name}")
            sys.exit(1)
    else:
        print("No model directories found in rl/results/ppo_models")
        sys.exit(1)
else:
    print("Results directory not found. Please train a model first.")
    sys.exit(1)

pnl, trades, trade_costs = [], [], []
for i in range(N_PATHS):
    if i % 1000 == 0:
        print(f"Evaluating path {i}/{N_PATHS}")
    
    obs, _ = env.reset()
    done = False
    while not done:
        act, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, info = env.step(act)
    
    # Calculate payoff using the same approach as in the environment
    final_spot = env.path_S[-1]
    if hasattr(deriv, 'call') and hasattr(deriv, 'strike'):
        if deriv.call:
            # Call option payoff = max(S_T - K, 0)
            payoff = max(0, final_spot - deriv.strike)
        else:
            # Put option payoff = max(K - S_T, 0)
            payoff = max(0, deriv.strike - final_spot)
    else:
        payoff = 0.0
    
    final_pnl = env.cum_pnl - payoff
    pnl.append(final_pnl)
    trades.append(env.trade_count)  # Now using actual trade count
    trade_costs.append(env.total_trade_cost)

pnl = np.array(pnl)
trades = np.array(trades)
trade_costs = np.array(trade_costs)
cvar5 = pnl[pnl <= np.quantile(pnl, 0.05)].mean()

summary = {
    "model": "PPO_Band",
    "price": 0.0,                       # not priced here
    "pnl_mean": pnl.mean(),
    "pnl_std":  pnl.std(),
    "cvar5":    cvar5,
    "trades_mean": trades.mean(),
    "trades_std": trades.std(),
    "trades_median": np.median(trades),
    "trade_costs_mean": trade_costs.mean(),
    "abs_error_mean": np.abs(pnl).mean()
}

# Create results directory if it doesn't exist
results_csv_dir = pathlib.Path("rl/results/csv")
results_csv_dir.mkdir(parents=True, exist_ok=True)

pd.DataFrame([summary]).to_csv("rl/results/csv/ppo_eval.csv", index=False)
print("\n" + "="*50)
print("EVALUATION RESULTS")
print("="*50)
for key, value in summary.items():
    if isinstance(value, float):
        print(f"{key:20s}: {value:.6f}")
    else:
        print(f"{key:20s}: {value}")
print("="*50)

# Print some additional statistics
print(f"\nTrade distribution:")
print(f"Min trades: {trades.min()}")
print(f"Max trades: {trades.max()}")
print(f"25th percentile: {np.percentile(trades, 25):.1f}")
print(f"75th percentile: {np.percentile(trades, 75):.1f}")
print(f"95th percentile: {np.percentile(trades, 95):.1f}")
