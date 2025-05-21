# compare_baselines.py
"""Quick experiment runner to benchmark No-Transaction Band Network (NTBN)
against standard baselines.

Baselines implemented
---------------------
1. **Black-Scholes Delta Hedge (analytic)**
2. **Deep Hedger - MLP (no band)**
3. **No-Transaction Band Network (NTBN)**

Metrics collected
-----------------
- *Initial price* output by each hedger
- *Terminal hedge P&L* (mean and standard deviation)
- *Average # of re-hedges* per path (proxy for turnover)
- *Mean absolute hedging error* (|final portfolio value - payoff|)

The script prints a table of results and can optionally save a CSV / PNG plot.

Usage
-----
```bash
python compare_baselines.py               # run with defaults
python compare_baselines.py --paths 50000 # higher Monte-Carlo samples
```
"""
from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import List

import matplotlib.pyplot as plt
import torch
import sys
sys.path.append("../")
from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
from src.nn import BlackScholes, Hedger, MultiLayerPerceptron

# Local NTBN implementation (ntbtn.py in repo root)
from scripts.ntbtn import NoTransactionBandNet

# -----------------------------------------------------------------------------
# Helper dataclass to aggregate results
# -----------------------------------------------------------------------------

@dataclass
class HedgeStats:
    name: str
    price: float
    pnl_mean: float
    pnl_std: float
    trades_mean: float
    abs_error_mean: float

    def to_dict(self):
        return {
            "model": self.name,
            "price": self.price,
            "pnl_mean": self.pnl_mean,
            "pnl_std": self.pnl_std,
            "trades_mean": self.trades_mean,
            "abs_error_mean": self.abs_error_mean,
        }


# -----------------------------------------------------------------------------
# Core evaluation function
# -----------------------------------------------------------------------------

def organize_results_by_name(stats: List[HedgeStats], results: List[dict]) -> dict:
    """Organize results by model name for easier plotting and analysis.
    
    Args:
        stats (List[HedgeStats]): List of HedgeStats objects containing model names and summary statistics
        results (List[dict]): List of result dictionaries containing raw data (pnl, nb_trade, error)
        
    Returns:
        dict: Dictionary mapping model names to their complete results
    """
    return {
        stat.name: {
            "stats": stat,
            "pnl": result["pnl"],
            "nb_trade": result["nb_trade"],
            "error": result["error"]
        }
        for stat, result in zip(stats, results)
    }

def evaluate_hedger(model: torch.nn.Module, derivative, n_paths: int, n_epochs: int) -> tuple[HedgeStats, dict]:
    """Train **and** test a hedger, returning summary statistics.

    A fresh *Hedger* wrapper is created for each model.
    The function re-trains the model each call - tweak as needed.
    """
    hedger = Hedger(model, model.inputs())

    # === TRAIN ===
    # Skip training if n_epochs is 0 or model has no parameters
    has_params = any(p.requires_grad for p in model.parameters())
    if n_epochs > 0 and has_params:
        hedger.fit(derivative, n_paths=n_paths, n_epochs=n_epochs, verbose=False)

    # === TEST / MONTE‑CARLO ===
    derivative.simulate(n_paths=n_paths)
    pl = hedger.compute_pl(derivative)  # Get P&L
    hedge = hedger.compute_hedge(derivative)  # Get hedge ratios

    # Calculate number of trades by looking at changes in hedge ratios
    trades = (hedge[..., 1:] != hedge[..., :-1]).sum(dim=-1).sum(dim=-1)
    # Calculate absolute error
    abs_error = (pl - derivative.payoff()).abs()

    # Create result dict
    result = {
        "pnl": pl,
        "nb_trade": trades,
        "error": pl - derivative.payoff()
    }

    price = hedger.price(derivative, n_paths=n_paths)

    return HedgeStats(
        name=model.__class__.__name__,
        price=float(price.detach()),
        pnl_mean=float(pl.mean()),
        pnl_std=float(pl.std()),
        trades_mean=float(trades.float().mean()),
        abs_error_mean=float(abs_error.mean()),
    ), result


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Deep-hedging baseline comparison")
    parser.add_argument("--paths", type=int, default=20_000, help="MC paths for train & test")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for NN models")
    parser.add_argument("--save", action="store_true", help="Save csv and png plot")
    args = parser.parse_args()

    torch.manual_seed(42)

    # Market model – GBM with modest vol
    spot = BrownianStock(sigma=0.2, mu=0.0, cost=1e-4)
    derivative = EuropeanOption(spot, maturity=1.0, strike=1.0)  # atm option on normalized spot

    stats: list[HedgeStats] = []
    results: list[dict] = []

    # 1 Black‑Scholes analytic hedge (no learning)
    bs_model = BlackScholes(derivative)
    stat, result = evaluate_hedger(bs_model, derivative, args.paths, n_epochs=0)
    stats.append(stat)
    results.append(result)

    # 2 Deep hedger – vanilla MLP (no band)
    # Create a wrapper to ensure MLP has inputs method matching BS model
    class MLPWrapper(torch.nn.Module):
        def __init__(self, n_units=(64, 64)):
            super().__init__()
            self.mlp = MultiLayerPerceptron(out_features=1, n_units=n_units, n_layers=len(n_units))
            
        def forward(self, x):
            return self.mlp(x)
            
        def inputs(self):
            # Use same inputs as BlackScholes
            return bs_model.inputs()
    
    mlp_model = MLPWrapper()
    stat, result = evaluate_hedger(mlp_model, derivative, args.paths, args.epochs)
    # Update name to be more descriptive
    stat = HedgeStats(
        name="DeepHedger",
        price=stat.price,
        pnl_mean=stat.pnl_mean,
        pnl_std=stat.pnl_std,
        trades_mean=stat.trades_mean,
        abs_error_mean=stat.abs_error_mean,
    )
    stats.append(stat)
    results.append(result)

    # 3 No‑Transaction Band Network
    ntbn = NoTransactionBandNet(derivative)
    stat, result = evaluate_hedger(ntbn, derivative, args.paths, args.epochs)
    stats.append(stat)
    results.append(result)

    # Organize results by name for plotting
    result_by_name = organize_results_by_name(stats, results)

    # --- Display summary table ---
    print("\n=== Comparison Results ===")
    # Print header
    print(f"{'Model':<15} {'Price':<10} {'PnL Mean':<10} {'PnL Std':<10} {'Trades Mean':<12} {'Error Mean':<12}")
    print("-" * 70)
    # Print rows
    for s in stats:
        print(f"{s.name:<15} {s.price:<10.6f} {s.pnl_mean:<10.6f} {s.pnl_std:<10.6f} {s.trades_mean:<12.6f} {s.abs_error_mean:<12.6f}")

    # --- Optional save ---
    if args.save:
        out_dir = pathlib.Path("results")
        out_dir.mkdir(exist_ok=True)
        
        # Save CSV
        csv_path = out_dir / "baseline_comparison.csv"
        with open(csv_path, 'w') as f:
            f.write("model,price,pnl_mean,pnl_std,trades_mean,abs_error_mean\n")
            for s in stats:
                f.write(f"{s.name},{s.price},{s.pnl_mean},{s.pnl_std},{s.trades_mean},{s.abs_error_mean}\n")
        
        # Simple PnL histogram plot
        plt.figure()
        for s, color in zip(stats, [None, None, None]):
            plt.hist(result_by_name[s.name]["pnl"].detach().numpy(), bins=100, histtype="step", label=s.name)
        plt.legend()
        plt.xlabel("terminal P&L")
        plt.ylabel("frequency")
        plt.title("PnL distribution - baselines vs NTBN")
        plt.tight_layout()
        plt.savefig(out_dir / "pnl_hist.png", dpi=150)
        print(f"Saved CSV -> {csv_path}\nSaved plot -> {out_dir/'pnl_hist.png'}")


if __name__ == "__main__":
    main()
