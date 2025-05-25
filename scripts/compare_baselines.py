# compare_baselines.py
"""Quick experiment runner to benchmark No-Transaction Band Network (NTBN)
against standard baselines.

Baselines implemented
---------------------
0. **Naked Option (hold without hedging)**
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
import sys

# Add the project root to the Python path so we can import from src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
from src.nn import BlackScholes, Hedger, MultiLayerPerceptron
from scripts.spntbtn import SoftplusBandNet
from scripts.cvar import CVaRLoss

# Local NTBN implementation (ntbtn.py in repo root)
from scripts.ntbtn import NoTransactionBandNet

# Import all utility functions and classes from the helper module
from scripts.baseline_utils import (
    NakedOption, PPOBandNet, MLPWrapper, HedgeStats,
    organize_results_by_name, debug_csv_values, 
    calculate_confidence_intervals, calculate_kelly_ratio, calculate_sharpe_ratio,
    evaluate_hedger, STABLE_BASELINES_AVAILABLE
)


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

    # 0 Naked Option - simply hold the option (baseline)
    naked_model = NakedOption(derivative)
    stat, result = evaluate_hedger(naked_model, derivative, args.paths, n_epochs=0)
    stat.name = "NakedOption"  # Override the class name for better display
    stats.append(stat)
    results.append(result)

    # 1 Black‑Scholes analytic hedge (no learning)
    bs_model = BlackScholes(derivative)
    stat, result = evaluate_hedger(bs_model, derivative, args.paths, n_epochs=0)
    stats.append(stat)
    results.append(result)

    # 2 Deep hedger – vanilla MLP (no band)
    mlp_model = MLPWrapper(derivative)
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
        kelly_ratio=stat.kelly_ratio,
        sharpe_ratio=stat.sharpe_ratio,
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
    print(f"{'Model':<20} {'Price':<10} {'PnL Mean':<10} {'PnL Std':<10} {'Trades Mean':<12} {'Error Mean':<12} {'CVaR 5%':<10} {'Kelly':<10} {'Sharpe':<10}")
    print("-" * 110)
    # Print rows
    for s in stats:
        print(f"{s.name:<20} {s.price:<10.6f} {s.pnl_mean:<10.6f} {s.pnl_std:<10.6f} {s.trades_mean:<12.6f} {s.abs_error_mean:<12.6f} {s.cvar5:<10.6f} {s.kelly_ratio:<10.6f} {s.sharpe_ratio:<10.6f}")

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
                f.write("model,price,pnl_mean,pnl_std,trades_mean,abs_error_mean,cvar5,kelly_ratio,sharpe_ratio\n")
                
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
                        kelly_ratio = str(float(s.kelly_ratio) if hasattr(s, 'kelly_ratio') and s.kelly_ratio is not None else 0.0)
                        sharpe_ratio = str(float(s.sharpe_ratio) if hasattr(s, 'sharpe_ratio') and s.sharpe_ratio is not None else 0.0)
                        
                        # Write the row
                        f.write(f"{name},{price},{pnl_mean},{pnl_std},{trades_mean},{abs_error_mean},{cvar5},{kelly_ratio},{sharpe_ratio}\n")
                    except Exception as row_err:
                        print(f"Warning: Could not write row for {s.name}: {row_err}")
                        # Write a row with placeholder values
                        f.write(f"{s.name},0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0\n")
            
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
            
            # Prepare raw data for confidence interval calculations
            # For metrics that don't have direct raw data, we'll use bootstrap from PnL
            pnl_data_list = []
            trade_data_list = []
            error_data_list = []
            
            for s in stats:
                model_results = result_by_name[s.name]
                pnl_data_list.append(model_results["pnl"])
                trade_data_list.append(model_results["nb_trade"])
                error_data_list.append(model_results["error"])
            
            # Calculate confidence intervals for each metric type
            pnl_mean_ci = calculate_confidence_intervals(pnl_data_list)
            trade_mean_ci = calculate_confidence_intervals(trade_data_list)
            error_mean_ci = calculate_confidence_intervals(error_data_list)
            
            # For price and PnL std, we'll use bootstrap sampling from PnL data
            price_ci = []
            pnl_std_ci = []
            
            for i, s in enumerate(stats):
                model_results = result_by_name[s.name]
                pnl_data = model_results["pnl"]
                
                # Convert to numpy
                if hasattr(pnl_data, 'cpu'):
                    pnl_np = pnl_data.cpu().numpy()
                elif hasattr(pnl_data, 'numpy'):
                    pnl_np = pnl_data.numpy()
                else:
                    pnl_np = np.array(pnl_data)
                
                # For price, we'll use a small dummy error since it's essentially deterministic
                # In practice, the "price" is the mean P&L, so we can use the P&L mean CI
                price_ci.append(pnl_mean_ci[i])
                
                # Bootstrap for PnL std confidence interval
                n_bootstrap = 1000
                bootstrap_stds = []
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(pnl_np, size=len(pnl_np), replace=True)
                    bootstrap_stds.append(np.std(bootstrap_sample))
                
                # Calculate confidence intervals for standard deviation
                std_mean = np.mean(bootstrap_stds)
                std_margin = np.percentile(bootstrap_stds, 97.5) - std_mean
                pnl_std_ci.append((std_mean, std_margin))
            
            # Convert all metrics to plain Python floats and prepare error bars
            metrics_data = {
                "Initial Price": {
                    "values": [float(s.price) for s in stats],
                    "errors": [ci[1] for ci in price_ci]
                },
                "Average P&L": {
                    "values": [float(s.pnl_mean) for s in stats],
                    "errors": [ci[1] for ci in pnl_mean_ci]
                },
                "P&L σ": {
                    "values": [float(s.pnl_std) for s in stats],
                    "errors": [ci[1] for ci in pnl_std_ci]
                },
                "Avg Trades": {
                    "values": [float(s.trades_mean) for s in stats],
                    "errors": [ci[1] for ci in trade_mean_ci]
                },
            }
            
            # Plot each metric with error bars
            for ax_i, (title, metric_data) in zip(ax.flatten(), metrics_data.items()):
                # Create x positions for the bars
                x_pos = np.arange(len(names))
                values = metric_data["values"]
                errors = metric_data["errors"]
                
                # Create bars with error bars
                bars = ax_i.bar(x_pos, values, color=colors, yerr=errors, 
                               capsize=5, ecolor='black', alpha=0.7)
                
                # Set ticks and labels properly to avoid warning
                ax_i.set_xticks(x_pos)
                ax_i.set_xticklabels(names, rotation=45, ha="right")
                ax_i.set_title(f"{title} (95% CI)")
                ax_i.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for i, (bar, value, error) in enumerate(zip(bars, values, errors)):
                    height = bar.get_height()
                    ax_i.text(bar.get_x() + bar.get_width()/2., height + error + (max(values) * 0.01),
                             f'{value:.3f}', ha='center', va='bottom', fontsize=8)
                             
        except Exception as e:
            print(f"Warning: Could not create metrics comparison plot: {e}")
            import traceback
            traceback.print_exc()
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
