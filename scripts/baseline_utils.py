"""
Utility functions and classes for baseline hedging strategy comparison.

This module contains:
- Model implementations (NakedOption, PPOBandNet, MLPWrapper)
- Evaluation functions
- Statistical calculation functions
- Data organization utilities
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import List, Tuple
import math
import sys
import os

import numpy as np
import torch

from src.nn import BlackScholes, Hedger, MultiLayerPerceptron

# PPO imports
try:
    from stable_baselines3 import PPO
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False


# -----------------------------------------------------------------------------
# Model Classes
# -----------------------------------------------------------------------------

class NakedOption(torch.nn.Module):
    """Naked option strategy - simply hold the option without hedging."""
    
    def __init__(self, derivative):
        super().__init__()
        self.derivative = derivative
        
    def inputs(self):
        """Return the same inputs as BlackScholes model for compatibility."""
        bs_model = BlackScholes(self.derivative)
        return bs_model.inputs()
    
    def forward(self, input_tensor):
        """
        Forward pass that returns zero hedge ratios (no hedging).
        
        Args:
            input_tensor: Tensor with shape (n_paths, n_steps, n_features)
        
        Returns:
            hedge_ratios: Tensor of zeros with shape (n_paths, n_steps, 1)
        """
        batch_size, n_steps, n_features = input_tensor.shape
        # Return zeros - no hedging at all
        return torch.zeros(batch_size, n_steps, 1)


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


class MLPWrapper(torch.nn.Module):
    """Wrapper for MLP to ensure compatibility with hedging framework."""
    
    def __init__(self, derivative, n_units=(64, 64)):
        super().__init__()
        self.derivative = derivative
        self.mlp = MultiLayerPerceptron(out_features=1, n_units=n_units, n_layers=len(n_units))
        
    def forward(self, x):
        return self.mlp(x)
        
    def inputs(self):
        # Use same inputs as BlackScholes
        bs_model = BlackScholes(self.derivative)
        return bs_model.inputs()


# -----------------------------------------------------------------------------
# Data Classes
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
    kelly_ratio: float = 0.0  # Kelly criterion ratio
    sharpe_ratio: float = 0.0  # Sharpe ratio

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
            "kelly_ratio": to_python_type(self.kelly_ratio),
            "sharpe_ratio": to_python_type(self.sharpe_ratio),
        }


# -----------------------------------------------------------------------------
# Utility Functions
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
        if hasattr(s, 'kelly_ratio'):
            print(f"  kelly_ratio: {type(s.kelly_ratio)} = {s.kelly_ratio}")
        else:
            print("  kelly_ratio: Not available")
        if hasattr(s, 'sharpe_ratio'):
            print(f"  sharpe_ratio: {type(s.sharpe_ratio)} = {s.sharpe_ratio}")
        else:
            print("  sharpe_ratio: Not available")


# -----------------------------------------------------------------------------
# Statistical Calculation Functions
# -----------------------------------------------------------------------------

def calculate_confidence_intervals(data_list: List[torch.Tensor], confidence_level: float = 0.95) -> List[Tuple[float, float]]:
    """Calculate confidence intervals for a list of data arrays using numpy only.
    
    Args:
        data_list: List of torch tensors containing the raw data for each model
        confidence_level: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        List of tuples containing (mean, margin_of_error) for each dataset
    """
    confidence_intervals = []
    alpha = 1 - confidence_level
    
    for data in data_list:
        # Convert to numpy array
        if hasattr(data, 'cpu'):
            data_np = data.cpu().numpy()
        elif hasattr(data, 'numpy'):
            data_np = data.numpy()
        else:
            data_np = np.array(data)
        
        # Calculate sample statistics
        mean = np.mean(data_np)
        std_error = np.std(data_np, ddof=1) / np.sqrt(len(data_np))  # Standard error of the mean
        
        # For large samples (n > 30), use normal approximation
        # For smaller samples, use t-distribution approximation
        n = len(data_np)
        if n > 30:
            # Use normal distribution (z-score)
            z_critical = 1.96  # For 95% confidence interval
            margin_of_error = z_critical * std_error
        else:
            # Use t-distribution approximation
            # For 95% CI and small samples, use conservative estimate
            t_critical = 2.0 + (0.5 / np.sqrt(n))  # Rough approximation
            margin_of_error = t_critical * std_error
        
        confidence_intervals.append((mean, margin_of_error))
    
    return confidence_intervals


def calculate_kelly_ratio(pnl_data: torch.Tensor) -> float:
    """Calculate Kelly criterion ratio for hedging strategy.
    
    Kelly ratio = Expected Return / Variance of Returns
    This gives the optimal fraction of capital to allocate.
    
    Args:
        pnl_data: Tensor containing P&L values
        
    Returns:
        Kelly ratio as float
    """
    # Convert to numpy with proper detaching
    if hasattr(pnl_data, 'detach'):
        pnl_data = pnl_data.detach()
    if hasattr(pnl_data, 'cpu'):
        pnl_np = pnl_data.cpu().numpy()
    elif hasattr(pnl_data, 'numpy'):
        pnl_np = pnl_data.numpy()
    else:
        pnl_np = np.array(pnl_data)
    
    mean_return = np.mean(pnl_np)
    variance = np.var(pnl_np, ddof=1)
    
    # Avoid division by zero
    if variance == 0:
        return 0.0
    
    kelly = mean_return / variance
    return float(kelly)


def calculate_sharpe_ratio(pnl_data: torch.Tensor, risk_free_rate: float = 0.0) -> float:
    """Calculate Sharpe ratio for hedging strategy.
    
    Sharpe ratio = (Expected Return - Risk-Free Rate) / Standard Deviation
    
    Args:
        pnl_data: Tensor containing P&L values
        risk_free_rate: Risk-free rate (default 0.0)
        
    Returns:
        Sharpe ratio as float
    """
    # Convert to numpy with proper detaching
    if hasattr(pnl_data, 'detach'):
        pnl_data = pnl_data.detach()
    if hasattr(pnl_data, 'cpu'):
        pnl_np = pnl_data.cpu().numpy()
    elif hasattr(pnl_data, 'numpy'):
        pnl_np = pnl_data.numpy()
    else:
        pnl_np = np.array(pnl_data)
    
    mean_return = np.mean(pnl_np)
    std_return = np.std(pnl_np, ddof=1)
    
    # Avoid division by zero
    if std_return == 0:
        return 0.0
    
    sharpe = (mean_return - risk_free_rate) / std_return
    return float(sharpe)


# -----------------------------------------------------------------------------
# Core Evaluation Function
# -----------------------------------------------------------------------------

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
    
    # Calculate Kelly ratio and Sharpe ratio
    kelly_ratio = calculate_kelly_ratio(pl)
    sharpe_ratio = calculate_sharpe_ratio(pl)
    
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
        kelly_ratio=kelly_ratio,
        sharpe_ratio=sharpe_ratio,
    ), result 