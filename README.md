# Deep Hedging System Improvements

This document outlines the key improvements made to the deep hedging system for better PPO performance and fair model comparison.

## Background

This work was inspired by the findings of Imaki et al. and their deep hedging framework. Building upon their foundation, I added the PPO (Proximal Policy Optimization) method for reinforcement learning-based hedging, along with enhanced implementations of the Softplus No-Transaction Band Network (spntbtn) and No-Transaction Band Network (ntbtn) methods.

## Overview of Improvements

### 1. PPO Fat Tail Elimination
**Problem**: PPO model exhibited fat-tailed P&L distributions due to gambling behavior.
**Solution**: Modified reward function with upside cap and variance penalty.

**Key Changes**:
- Reward capped at 0 when P&L > shortfall threshold (-10%)
- Added variance penalty (1e-3 coefficient) to discourage high-variance strategies
- Implemented P&L history tracking for variance calculation

**Files Modified**:
- `rl/env/hedging_env.py` - Enhanced reward function
- `rl/experiments/train_ppo_improved.py` - New training script with improved setup

### 2. Trade Frequency Analysis & Normalization
**Problem**: Confusion about PPO having different trade frequencies than other models.
**Solution**: Added analysis tools and optional trade normalization.

**Key Insights**:
- PPO naturally trades less (~20-80 trades) vs. BlackScholes (~240 trades)
- This is the **correct behavior** - fewer trades = lower transaction costs
- Different strategies naturally have different trade frequencies

**New Features**:
- `SimpleBandWrapper` class to apply no-transaction bands to any model
- `--normalize-trades` flag for fair trade comparison (optional)
- Enhanced trade counting with proper floating-point precision

### 3. Histogram Visualization Fix
**Problem**: Naked option showed much higher frequencies in histograms due to binning issues.
**Solution**: Fixed histogram plotting with proper binning and density normalization.

**Root Cause**: 
- Naked option has ~58% of values at exactly P&L=0 (out-of-money options)
- Different data ranges caused different bin widths
- Created misleading "frequency" differences

**Fix**: 
- Common bin range across all models
- Added density normalization plot
- Side-by-side raw counts vs. normalized density

## Quick Start

### 1. Train Improved PPO Model
```bash
python rl/experiments/train_ppo_improved.py
```

### 2. Run Baseline Comparison
```bash
# Standard comparison (natural trading strategies)
python scripts/compare_baselines.py --paths 20000 --save

# Trade-normalized comparison (all models use bands)
python scripts/compare_baselines.py --paths 20000 --save --normalize-trades --band-width 0.02
```

### 3. View Results
Results saved to `results/` directory:
- `baseline_comparison.csv` - Summary statistics
- `pnl_hist.png` - P&L distributions (fixed visualization)
- `metrics_comparison.png` - Performance metrics with confidence intervals
- `trade_vol_scatter.png` - Risk vs. turnover analysis

## Key Files

### Core Improvements
- `rl/env/hedging_env.py` - Enhanced reward function
- `scripts/baseline_utils.py` - SimpleBandWrapper, improved model integration
- `scripts/compare_baselines.py` - Fixed histograms, trade normalization

### Training & Evaluation
- `rl/experiments/train_ppo_improved.py` - Improved PPO training
- `scripts/ntbtn.py` - No-transaction band network
- `scripts/spntbtn.py` - Softplus band network

## Understanding the Results

### Trade Frequency Interpretation
- **Low trades (PPO, NTBN)**: Band-based strategies, lower transaction costs
- **High trades (BlackScholes, DeepHedger)**: Delta hedging, higher transaction costs
- **This difference is feature, not bug**: Different strategies have different natural behaviors

### P&L Distribution Analysis
- **Raw counts**: Shows actual frequency distribution
- **Density**: Normalized for fair shape comparison
- **Naked option spike at 0**: Expected behavior for out-of-money options

### Performance Metrics
- **CVaR 5%**: Tail risk measure (lower = better)
- **Kelly Ratio**: Risk-adjusted return metric
- **Sharpe Ratio**: Risk-adjusted performance
- **Confidence intervals**: Statistical significance of differences

## Next Steps

1. **Train longer**: Increase training epochs for better convergence
2. **Parameter tuning**: Experiment with different band widths, reward weights
3. **Market conditions**: Test on different volatility regimes, maturities
4. **Alternative objectives**: Try different risk measures (VaR, ES, etc.)

## Dependencies

```bash
pip install torch numpy matplotlib stable-baselines3
# For RL training: gymnasium, tensorboard
```

## Notes

- Use `--normalize-trades` only for academic comparison
- Natural trading strategies show real-world advantages
- PPO models are automatically prioritized (improved > regular)
- Fixed histogram binning eliminates visual artifacts 