# compare_baselines.py
"""Quick experiment runner to benchmark No-Transaction Band Network (NTBN)
against standard baselines.

Baselines implemented
---------------------
1. **Black-Scholes Delta Hedge (analytic)**
2. **Deep Hedger - MLP (no band)**
3. **No-Transaction Band Network (NTBN)**
4. **PPO Band Network (RL)**

Metrics collected
-----------------
- *Initial price* output by each hedger
- *Terminal hedge P&L* (mean and standard deviation)
- *Average # of re-hedges* per path (proxy for turnover)
- *Mean absolute hedging error* (|final portfolio value - payoff|)

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
import math
import sys
import os

# Add the project root to the Python path so we can import from src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
import torch

from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
from src.nn import BlackScholes, Hedger, MultiLayerPerceptron
from scripts.spntbtn import SoftplusBandNet
from scripts.cvar import CVaRLoss

# Local NTBN implementation (ntbtn.py in repo root)
from scripts.ntbtn import NoTransactionBandNet

# PPO imports
try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False
    print("Warning: stable-baselines3 not available. PPO model will be skipped.")


# -----------------------------------------------------------------------------
# PPO Wrapper for RL-based hedging
# -----------------------------------------------------------------------------

class PPOBandNet(torch.nn.Module):
    """Wrapper for trained PPO model that implements no-transaction band hedging."""
    
    def __init__(self, derivative, model_path: str = None):
        super().__init__()
        self.derivative = derivative
        self.model_path = model_path
        self.ppo_model = None
        self.cost = 1e-4
        self.prev_hedge = 0.0
        
        # Load the trained PPO model
        if model_path and STABLE_BASELINES_AVAILABLE:
            try:
                self.ppo_model = PPO.load(model_path)
                print(f"Loaded PPO model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load PPO model from {model_path}: {e}")
                self.ppo_model = None
        else:
            # Try to find the most recent model
            self._find_latest_model()
    
    def _find_latest_model(self):
        """Find the most recent trained PPO model."""
        try:
            results_dir = pathlib.Path("rl/results/ppo_models")
            if results_dir.exists():
                model_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
                if model_dirs:
                    latest_model_dir = max(model_dirs, key=lambda x: x.stat().st_mtime)
                    model_path = latest_model_dir / "ppo_band_final.zip"
                    if model_path.exists():
                        self.ppo_model = PPO.load(str(model_path))
                        print(f"Auto-loaded PPO model from {model_path}")
                    else:
                        print("PPO model file not found")
        except Exception as e:
            print(f"Could not auto-load PPO model: {e}")
    
    def inputs(self):
        """Return the same inputs as BlackScholes model for compatibility."""
        bs_model = BlackScholes(self.derivative)
        return bs_model.inputs()
    
    def _normal_cdf(self, x):
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))
    
    def _calculate_bs_delta(self, spot, time_to_maturity):
        """Calculate Black-Scholes delta for the option."""
        if time_to_maturity <= 0:
            # At expiry
            if hasattr(self.derivative, 'call') and self.derivative.call:
                return 1.0 if spot > self.derivative.strike else 0.0
            else:
                return -1.0 if spot < self.derivative.strike else 0.0
        
        # Standard BS delta calculation
        volatility = 0.2  # Should match the training environment
        strike = self.derivative.strike
        
        d1 = (np.log(spot / strike) + (0.5 * volatility**2) * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        
        if hasattr(self.derivative, 'call') and self.derivative.call:
            return self._normal_cdf(d1)
        else:
            return self._normal_cdf(d1) - 1.0
    
    def forward(self, input_tensor):
        """
        Forward pass that computes hedge ratios using PPO policy.
        
        Args:
            input_tensor: Tensor with shape (n_paths, n_steps, n_features)
                         Features: [log_moneyness, time_to_maturity, volatility, ...]
        
        Returns:
            hedge_ratios: Tensor with shape (n_paths, n_steps, 1)
        """
        if self.ppo_model is None:
            # Fallback to BS delta if no PPO model available
            log_moneyness = input_tensor[..., 0]
            time_to_maturity = input_tensor[..., 1]
            spot = torch.exp(log_moneyness) * self.derivative.strike
            
            # Calculate BS delta for each point
            hedge_ratios = torch.zeros_like(input_tensor[..., :1])
            for i in range(input_tensor.shape[0]):
                for j in range(input_tensor.shape[1]):
                    s = spot[i, j].item()
                    tau = time_to_maturity[i, j].item()
                    delta = self._calculate_bs_delta(s, tau)
                    hedge_ratios[i, j, 0] = delta
            
            return hedge_ratios
        
        # Use PPO model for hedging decisions
        batch_size, n_steps, n_features = input_tensor.shape
        hedge_ratios = torch.zeros(batch_size, n_steps, 1)
        
        for path_idx in range(batch_size):
            self.prev_hedge = 0.0  # Reset for each path
            
            for step_idx in range(n_steps):
                # Extract current state
                log_moneyness = input_tensor[path_idx, step_idx, 0].item()
                time_to_maturity = input_tensor[path_idx, step_idx, 1].item()
                volatility = input_tensor[path_idx, step_idx, 2].item() if n_features > 2 else 0.2
                
                # Current spot price
                spot = np.exp(log_moneyness) * self.derivative.strike
                
                # Calculate BS delta
                bs_delta = self._calculate_bs_delta(spot, time_to_maturity)
                
                # Create observation for PPO (matching training environment format)
                # [log_moneyness, time_to_maturity, volatility, prev_hedge, cum_pnl, trade_count]
                obs = np.array([
                    log_moneyness,
                    time_to_maturity, 
                    volatility,
                    self.prev_hedge,
                    0.0,  # cum_pnl (not used for action selection)
                    0.0   # trade_count (not used for action selection)
                ], dtype=np.float32)
                
                # Get action from PPO policy
                try:
                    action, _ = self.ppo_model.predict(obs, deterministic=True)
                    
                    # Apply softplus to get band widths
                    widths = torch.nn.functional.softplus(torch.tensor(action)).numpy()
                    
                    # Apply no-transaction band logic
                    band_min = bs_delta - widths[0]
                    band_max = bs_delta + widths[1]
                    
                    # Clamp current hedge to band
                    new_hedge = np.clip(self.prev_hedge, band_min, band_max)
                    self.prev_hedge = new_hedge
                    
                    hedge_ratios[path_idx, step_idx, 0] = new_hedge
                    
                except Exception as e:
                    print(f"PPO prediction error: {e}, using BS delta")
                    hedge_ratios[path_idx, step_idx, 0] = bs_delta
                    self.prev_hedge = bs_delta
        
        return hedge_ratios


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
    cvar5: float = 0.0  # Conditional Value at Risk (5%)

    def to_dict(self):
        """Ensure all values are native Python types, not tensors or numpy arrays"""
        def to_python_type(val):
            if hasattr(val, 'item'):
                return val.item()  # For PyTorch tensors
            elif hasattr(val, 'tolist'):
                return val.tolist()  # For numpy arrays
            return val  # Already a Python type
            
        return {
            "model": str(self.name),
            "price": to_python_type(self.price),
            "pnl_mean": to_python_type(self.pnl_mean),
            "pnl_std": to_python_type(self.pnl_std),
            "trades_mean": to_python_type(self.trades_mean),
            "abs_error_mean": to_python_type(self.abs_error_mean),
            "cvar5": to_python_type(self.cvar5),
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

def evaluate_hedger(model: torch.nn.Module, derivative, n_paths: int, n_epochs: int, loss: torch.nn.Module = None) -> tuple[HedgeStats, dict]:
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

    # Calculate CVaR at 5% level using PyTorch
    q = torch.quantile(pl, 0.05)
    cvar5 = pl[pl <= q].mean().item()

    # Get the price as a Python float
    price = hedger.price(derivative, n_paths=n_paths).item()
    
    # Create result dict with detached tensors
    result = {
        "pnl": pl.detach(),
        "nb_trade": trades.detach(),
        "error": (pl - derivative.payoff()).detach()
    }

    return HedgeStats(
        name=model.__class__.__name__,
        price=price,  # Already converted to item() above
        pnl_mean=pl.mean().item(),
        pnl_std=pl.std().item(),
        trades_mean=trades.float().mean().item(),
        abs_error_mean=abs_error.mean().item(),
        cvar5=cvar5,  # Already converted to item() above
    ), result


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------

def debug_csv_values(stats: List[HedgeStats]) -> None:
    """Print debug information for CSV values."""
    for i, s in enumerate(stats):
        print(f"\nModel {i+1}: {s.name}")
        print(f"  price: {type(s.price)} = {s.price}")
        print(f"  pnl_mean: {type(s.pnl_mean)} = {s.pnl_mean}")
        print(f"  pnl_std: {type(s.pnl_std)} = {s.pnl_std}")
        print(f"  trades_mean: {type(s.trades_mean)} = {s.trades_mean}")
        print(f"  abs_error_mean: {type(s.abs_error_mean)} = {s.abs_error_mean}")
        if hasattr(s, 'cvar5'):
            print(f"  cvar5: {type(s.cvar5)} = {s.cvar5}")
        else:
            print("  cvar5: Not available")
            

def main():
    parser = argparse.ArgumentParser(description="Deep-hedging baseline comparison")
    parser.add_argument("--paths", type=int, default=20_000, help="MC paths for train & test")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for NN models")
    parser.add_argument("--save", action="store_true", help="Save csv and png plot")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--ppo-model", type=str, default=None, help="Path to trained PPO model (optional)")
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
        cvar5=stat.cvar5,
    )
    stats.append(stat)
    results.append(result)

    # 3 No‑Transaction Band Network
    ntbn = NoTransactionBandNet(derivative)
    stat, result = evaluate_hedger(ntbn, derivative, args.paths, args.epochs)
    stats.append(stat)
    results.append(result)

    # 4 Softplus Band Network
    sp_ntbn = SoftplusBandNet(derivative)
    stat, result = evaluate_hedger(sp_ntbn, derivative, args.paths, args.epochs)
    stats.append(stat)
    results.append(result)

    # 5 Softplus NTBN + CVaR
    sp_ntbn_cvar = SoftplusBandNet(derivative)
    stat, result = evaluate_hedger(sp_ntbn_cvar, derivative, args.paths, args.epochs, loss=CVaRLoss(alpha=0.05, cost_weight=1e-4))
    stat.name = "SoftplusBandNet_CVaR"
    stats.append(stat)
    results.append(result)

    # 6 PPO Band Network (RL-based)
    if STABLE_BASELINES_AVAILABLE:
        try:
            print("Loading PPO model...")
            ppo_model = PPOBandNet(derivative, model_path=args.ppo_model)
            if ppo_model.ppo_model is not None:  # Only evaluate if model was loaded successfully
                print("Evaluating PPO model...")
                stat, result = evaluate_hedger(ppo_model, derivative, args.paths, n_epochs=0)  # No training needed
                stat.name = "PPO_BandNet"
                stats.append(stat)
                results.append(result)
                print(f"✓ PPO model successfully integrated into comparison (#{len(stats)})")
            else:
                print("⚠ PPO model not found or failed to load - skipping from comparison")
                print("  To include PPO: train a model first with 'python rl/experiments/train_ppo.py'")
        except Exception as e:
            print(f"✗ Error integrating PPO model: {e}")
            print("  Make sure you have trained a PPO model first")
    else:
        print("⚠ Stable Baselines 3 not available - skipping PPO model")
        print("  Install with: pip install stable-baselines3")

    # Organize results by name for plotting
    result_by_name = organize_results_by_name(stats, results)

    # --- Display summary table ---
    print(f"\n=== Comparison Results ({len(stats)} models) ===")
    # Print header
    print(f"{'Model':<20} {'Price':<10} {'PnL Mean':<10} {'PnL Std':<10} {'Trades Mean':<12} {'Error Mean':<12} {'CVaR 5%':<10}")
    print("-" * 90)
    # Print rows
    for s in stats:
        print(f"{s.name:<20} {s.price:<10.6f} {s.pnl_mean:<10.6f} {s.pnl_std:<10.6f} {s.trades_mean:<12.6f} {s.abs_error_mean:<12.6f} {s.cvar5:<10.6f}")

    # Highlight PPO if included
    ppo_included = any("PPO" in s.name for s in stats)
    if ppo_included:
        print(f"\n✓ PPO model included in all visualizations and analysis")
    else:
        print(f"\n⚠ PPO model not included - see messages above")

    # Print debug information if requested
    if args.debug:
        debug_csv_values(stats)

    # --- Optional save ---
    if args.save:
        out = pathlib.Path("results")
        out.mkdir(exist_ok=True)
        
        # CSV - write directly to file without pandas
        try:
            # Define the CSV file path
            csv_path = out / "baseline_comparison.csv"
            
            # Open the file and write directly
            with open(csv_path, 'w') as f:
                # Write header
                f.write("model,price,pnl_mean,pnl_std,trades_mean,abs_error_mean,cvar5\n")
                
                # Write each row
                for s in stats:
                    try:
                        # Safely convert each value to string representation of float
                        name = str(s.name)
                        price = str(float(s.price) if hasattr(s.price, 'item') else float(s.price))
                        pnl_mean = str(float(s.pnl_mean) if hasattr(s.pnl_mean, 'item') else float(s.pnl_mean))
                        pnl_std = str(float(s.pnl_std) if hasattr(s.pnl_std, 'item') else float(s.pnl_std))
                        trades_mean = str(float(s.trades_mean) if hasattr(s.trades_mean, 'item') else float(s.trades_mean))
                        abs_error_mean = str(float(s.abs_error_mean) if hasattr(s.abs_error_mean, 'item') else float(s.abs_error_mean))
                        cvar5 = str(float(s.cvar5) if hasattr(s, 'cvar5') and s.cvar5 is not None else 0.0)
                        
                        # Write the row
                        f.write(f"{name},{price},{pnl_mean},{pnl_std},{trades_mean},{abs_error_mean},{cvar5}\n")
                    except Exception as row_err:
                        print(f"Warning: Could not write row for {s.name}: {row_err}")
                        # Write a row with placeholder values
                        f.write(f"{s.name},0.0,0.0,0.0,0.0,0.0,0.0\n")
            
            print(f"CSV saved successfully to {csv_path}")
            
        except Exception as e:
            print(f"Warning: Could not save CSV due to: {e}")

        # Histograms
        try:
            cmap = plt.get_cmap('tab10')
            colors = cmap(np.linspace(0, 1, len(stats)))  # Auto-adapt colors to number of models (including PPO)
            plt.figure(figsize=(10, 6))
            for i, s in enumerate(stats):
                # Safely convert to numpy array with error handling
                try:
                    pnl = result_by_name[s.name]["pnl"]
                    # Handle different tensor types
                    if hasattr(pnl, 'cpu'):
                        pnl_data = pnl.cpu().numpy()
                    elif hasattr(pnl, 'numpy'):
                        pnl_data = pnl.numpy()
                    else:
                        pnl_data = np.array(pnl)
                    
                    plt.hist(pnl_data, bins=60, histtype="step", lw=2, color=colors[i], label=s.name)
                except Exception as e:
                    print(f"Warning: Could not plot histogram for {s.name}: {e}")
                    
            plt.axvline(0, ls="--", c="k", alpha=0.4)
            plt.xlabel("Terminal P&L"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout()
            plt.savefig(out / "pnl_hist.png", dpi=150)
            print(f"Saved P&L histogram with {len(stats)} models")
        except Exception as e:
            print(f"Warning: Could not create histogram plot: {e}")

        # Metric bar charts (Price, Mean P&L, σ, Trades) - includes all models
        try:
            fig, ax = plt.subplots(2, 2, figsize=(12, 9)); fig.suptitle("Model Performance Comparison")
            names = [s.name for s in stats]
            # Convert all metrics to plain Python floats
            metrics = {
                "Initial Price": [float(s.price) for s in stats],
                "Average P&L": [float(s.pnl_mean) for s in stats],
                "P&L σ": [float(s.pnl_std) for s in stats],
                "Avg Trades": [float(s.trades_mean) for s in stats],
            }
            
            # Plot each metric
            for ax_i, (title, values) in zip(ax.flatten(), metrics.items()):
                # Create x positions for the bars
                x_pos = np.arange(len(names))
                ax_i.bar(x_pos, values, color=colors)
                
                # Set ticks and labels properly to avoid warning
                ax_i.set_xticks(x_pos)
                ax_i.set_xticklabels(names, rotation=45, ha="right")
                ax_i.set_title(title)
                ax_i.grid(axis='y', alpha=0.3)
        except Exception as e:
            print(f"Warning: Could not create metrics comparison plot: {e}")
        plt.tight_layout(); plt.subplots_adjust(top=0.92)
        plt.savefig(out / "metrics_comparison.png", dpi=150)

        # Risk‑turnover scatter (σ & CVaR)
        try:
            plt.figure(figsize=(6, 5))
            for i, s in enumerate(stats):
                # Convert tensor values to Python floats
                trades = float(s.trades_mean)
                pnl_std = float(s.pnl_std)
                
                # Plot scatter point
                plt.scatter(trades, pnl_std, color=colors[i], label=s.name, marker='o', s=80)
                
                # Add label to each point
                model_name = s.name.split('_')[0] if '_' in s.name else s.name[:10]
                plt.text(trades, pnl_std, model_name, fontsize=7, ha='right')
                
            plt.xlabel("Average # Trades"); plt.ylabel("P&L σ"); plt.title("Turnover vs Volatility")
            plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out / "trade_vol_scatter.png", dpi=150)
        except Exception as e:
            print(f"Warning: Could not create trade-volatility scatter plot: {e}")

        # CVaR scatter (optional – on same figure)
        try:
            plt.figure(figsize=(6, 5))
            for i, s in enumerate(stats):
                # Convert tensor values to Python floats
                trades = float(s.trades_mean)
                cvar = float(s.cvar5) if hasattr(s, 'cvar5') else 0.0
                
                # Plot scatter point
                plt.scatter(trades, cvar, color=colors[i], label=s.name, marker='o', s=80)
                
                # Add label to each point
                model_name = s.name.split('_')[0] if '_' in s.name else s.name[:10]
                plt.text(trades, cvar, model_name, fontsize=7, ha='right')
                
            plt.xlabel("Average # Trades"); plt.ylabel("CVaR 5% (lower = better)"); plt.title("Turnover vs CVaR")
            plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(out / "trade_cvar_scatter.png", dpi=150)
        except Exception as e:
            print(f"Warning: Could not create trade-CVaR scatter plot: {e}")

        print("Saved outputs to", out.resolve())


if __name__ == "__main__":
    main()
