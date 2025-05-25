# PPO Fat Tail Fix: Improved Reward Function

## Problem Identification

The original PPO model was exhibiting **fat tails** in its P&L distribution, meaning it had a higher probability of extreme gains or losses than expected. This is problematic for hedging because:

1. **Gambling behavior**: The model was incentivized to take excessive risks for potential high returns
2. **Unstable hedging**: High variance strategies are unsuitable for risk management
3. **No upside cap**: The reward function didn't limit rewards for good performance, encouraging risky strategies

## Root Cause

The original reward function in `rl/env/hedging_env.py` only penalized poor performance (shortfall losses) but provided unlimited rewards for good performance. This created an asymmetric incentive structure that encouraged the model to "gamble" for higher gains.

```python
# Original problematic reward
shortfall_penalty = -max(0.0, self.shortfall_thr - final_pnl)
reward = shortfall_penalty + trade_penalty
```

## Solution Implemented

### 1. Reward Capping
- **Cap reward at 0** when final P&L > shortfall threshold (-10%)
- This eliminates the incentive to gamble for excessive gains
- Focus shifts to consistent hedging rather than maximizing returns

```python
# New improved reward function
if final_pnl > self.shortfall_thr:
    # Good hedging - no additional reward to prevent gambling
    shortfall_reward = 0.0
else:
    # Poor hedging - penalize the shortfall
    shortfall_penalty = -max(0.0, self.shortfall_thr - final_pnl)
    shortfall_reward = shortfall_penalty
```

### 2. Variance Penalty
- **Track P&L history** across episodes
- **Penalize high variance** strategies to discourage fat tails
- Only applied when sufficient history is available (≥100 episodes)

```python
# Variance penalty to discourage fat tails
if len(self.pnl_history) >= 100:
    pnl_variance = np.var(self.pnl_history)
    variance_penalty = -1e-3 * max(0, pnl_variance - 0.01)  # Penalty if variance > 1%
```

### 3. Maintained Trade Efficiency
- **Keep trade cost penalties** to encourage efficient trading
- **Slightly increased penalty** to further discourage excessive trading

```python
trade_penalty = -1e-4 * self.trade_count  # Small penalty per trade
```

## Files Modified

### 1. `rl/env/hedging_env.py`
- Modified reward function to cap upside
- Added P&L history tracking for variance calculation
- Enhanced reward structure with variance penalty

### 2. `rl/experiments/train_ppo_improved.py` (NEW)
- New training script for improved PPO models
- Clear documentation of improvements
- Automatic model naming with "improved" tag

### 3. `scripts/baseline_utils.py`
- Updated `PPOBandNet` to prioritize improved models
- Automatic detection and loading of latest improved models
- Fallback to regular models if improved not available

## Usage Instructions

### Training Improved PPO Model

```bash
# Train with default settings (2M timesteps)
python rl/experiments/train_ppo_improved.py

# Longer training run
python rl/experiments/train_ppo_improved.py --timesteps 5_000_000

# Custom parameters
python rl/experiments/train_ppo_improved.py \
    --timesteps 3_000_000 \
    --shortfall -0.08 \
    --lr 2e-4 \
    --seed 123 \
    --tag no_fat_tails
```

### Comparing Results

```bash
# Compare all baselines (will automatically use improved PPO if available)
python scripts/compare_baselines.py --save

# Force use of specific improved model
python scripts/compare_baselines.py \
    --ppo-model rl/results/ppo_models/improved_20250101-120000/ppo_band_improved_final.zip \
    --save
```

### Evaluating Single Model

```bash
# Evaluate latest improved model
python rl/experiments/evaluate.py
```

## Expected Improvements

### 1. Reduced Tail Risk
- **Lower CVaR**: Better worst-case scenario protection
- **Reduced variance**: More consistent P&L distribution
- **Fewer extreme outliers**: Elimination of gambling behavior

### 2. More Stable Hedging
- **Consistent performance**: Less variation across different market scenarios
- **Predictable risk**: Better risk management characteristics
- **Professional hedging**: Behavior more aligned with institutional hedging practices

### 3. Better Risk-Adjusted Metrics
- **Improved Sharpe ratio**: Better risk-adjusted returns
- **Lower Kelly ratio**: More conservative position sizing
- **Reduced maximum drawdown**: Better downside protection

## Technical Details

### Reward Function Components

1. **Shortfall Component**: `shortfall_reward`
   - 0 if P&L > -10% (good hedging)
   - Negative penalty if P&L ≤ -10% (poor hedging)

2. **Trade Efficiency Component**: `trade_penalty`
   - Small negative penalty per trade
   - Encourages efficient rebalancing

3. **Variance Component**: `variance_penalty`
   - Penalizes strategies with variance > 1%
   - Only applied after sufficient training history

### Hyperparameter Tuning

Key parameters that can be adjusted:

- `shortfall_thr`: Threshold for good vs poor hedging (default: -0.10)
- Trade penalty coefficient: Currently `1e-4` per trade
- Variance penalty coefficient: Currently `1e-3` for excess variance
- Variance threshold: Currently 1% (0.01)
- History size: Currently 1000 episodes for variance calculation

## Monitoring Training Progress

### Tensorboard Logs
```bash
# View training progress
tensorboard --logdir rl/results/ppo_models/improved_*/tb
```

### Key Metrics to Watch
1. **Episode reward**: Should stabilize without extreme spikes
2. **P&L variance**: Should decrease over training
3. **Policy entropy**: Should remain reasonable for exploration
4. **Value function loss**: Should converge smoothly

## Troubleshooting

### Common Issues

1. **Slow convergence**: Try increasing learning rate or training timesteps
2. **Still seeing fat tails**: Increase variance penalty coefficient
3. **Too conservative**: Reduce trade penalty or variance penalty
4. **Model not loading**: Check path and ensure training completed successfully

### Performance Validation

Compare these metrics between old and improved models:
- CVaR (5%): Should be higher (less negative)
- P&L standard deviation: Should be lower
- Maximum loss: Should be reduced
- Sharpe ratio: Should be higher
- Kelly ratio: Should be lower (more conservative)

This improved reward structure should significantly reduce fat tails while maintaining effective hedging performance. 