import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import math

class HedgingEnv(gym.Env):
    """A simple hedging environment for reinforcement learning."""
    
    metadata = {"render_modes": []}

    def __init__(self, derivative, spot_model, steps=250, cost=1e-4, shortfall_thr=-0.10):
        super().__init__()
        self.derivative = derivative
        self.spot = spot_model
        self.n_steps = steps
        self.cost = cost
        self.shortfall_thr = shortfall_thr
        self.trade_count = 0

        # Action: 2 real numbers → any ℝ; later we softplus
        self.action_space = spaces.Box(low=-5., high=5., shape=(2,), dtype=np.float32)

        # Observation: 5 floats
        high = np.array([5, 1, 1, 1, 10, 250], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self.t = 0
        # Use simulate() instead of reset() for BrownianStock
        self.spot.simulate(n_paths=1, time_horizon=1.0)
        self.path_S = self.spot.spot.squeeze().numpy()   # shape (n_steps+1,)
        self.dt = 1 / self.n_steps
        self.prev_hedge = 0.0
        self.cum_pnl = 0.0
        self.trade_count = 0  # Track actual number of trades
        self.total_trade_cost = 0.0  # Track cumulative trading costs
        obs = self._get_obs()
        # Ensure observation is correct format
        info = {}
        return obs, info

    def step(self, action):
        # 1) Softplus so widths ≥ 0
        widths = torch.nn.functional.softplus(torch.tensor(action)).numpy()
        s_t = self.path_S[self.t]
        
        # 2) Compute Delta using Black-Scholes formula
        # Implementation of Black-Scholes delta for European call option
        tau = self._tau()
        if hasattr(self.derivative, 'call') and self.derivative.call:
            # Call option
            if tau > 0:
                volatility = self.spot.sigma
                strike = self.derivative.strike
                
                # Black-Scholes d1 calculation
                d1 = (np.log(s_t / strike) + (0.5 * volatility**2) * tau) / (volatility * np.sqrt(tau))
                # N(d1) is delta for a call option
                delta = self._normal_cdf(d1)
            else:
                # At expiry, delta is either 0 or 1
                delta = 1.0 if s_t > self.derivative.strike else 0.0
        else:
            # Put option or default fallback
            if tau > 0:
                volatility = self.spot.sigma
                strike = self.derivative.strike
                
                # Black-Scholes d1 calculation
                d1 = (np.log(s_t / strike) + (0.5 * volatility**2) * tau) / (volatility * np.sqrt(tau))
                # N(d1) - 1 is delta for a put option
                delta = self._normal_cdf(d1) - 1.0
            else:
                # At expiry, delta is either -1 or 0
                delta = -1.0 if s_t < self.derivative.strike else 0.0
        
        band_min = delta - widths[0]
        band_max = delta + widths[1]

        # 3) Clamp
        new_hedge = np.clip(self.prev_hedge, band_min, band_max)
        trade = new_hedge - self.prev_hedge
        
        # Count actual trades (when trade amount is non-trivial)
        if abs(trade) > 1e-6:  # Only count as trade if above small threshold
            self.trade_count += 1
        
        self.prev_hedge = new_hedge

        # 4) Cash-flow from hedge rebalancing
        dS = self.path_S[self.t+1] - s_t
        trade_cost = self.cost * abs(trade)
        self.total_trade_cost += trade_cost
        
        self.cum_pnl += -trade * s_t    
        self.cum_pnl += self.prev_hedge * dS           # hedge P&L

        self.t += 1
        terminated = self.t == self.n_steps

        if terminated:
            # Calculate final option payoff
            final_spot = self.path_S[-1]
            if hasattr(self.derivative, 'call') and hasattr(self.derivative, 'strike'):
                # Directly compute European option payoff
                if self.derivative.call:
                    # Call option payoff = max(S_T - K, 0)
                    payoff = max(0, final_spot - self.derivative.strike)
                else:
                    # Put option payoff = max(K - S_T, 0)
                    payoff = max(0, self.derivative.strike - final_spot)
            else:
                # Try to use the derivative's payoff function if available
                try:
                    # Update the underlier's spot price first
                    old_spot = self.derivative.underlier.spot
                    last_spot = torch.tensor([[final_spot]])
                    self.derivative.underlier.register_buffer("spot", last_spot)
                    
                    # Get the payoff from the derivative
                    payoff_tensor = self.derivative.payoff()
                    payoff = payoff_tensor.item()
                    
                    # Restore the original spot price
                    self.derivative.underlier.register_buffer("spot", old_spot)
                except Exception:
                    # Default to zero payoff if all else fails
                    payoff = 0.0
            
            final_pnl = self.cum_pnl - payoff
            # shortfall loss reward - penalize both shortfall and excessive trading
            shortfall_penalty = -max(0.0, self.shortfall_thr - final_pnl)
            
            # Add a small penalty for excessive trading (optional - you can tune this)
            trade_penalty = 5e-5 * self.trade_count  # Small penalty per trade
            
            reward = shortfall_penalty + trade_penalty
        else:
            # Optional: Small intermediate penalty for trading costs
            reward = -self.cost * abs(trade) * 0.1
        obs = self._get_obs()
        info = {
            'trade_count': self.trade_count,
            'total_trade_cost': self.total_trade_cost,
            'final_pnl': self.cum_pnl if terminated else None
        }
        return obs, reward, terminated, False, info

    # -------- helpers --------------------------------------------------------
    def _tau(self):
        return (self.n_steps - self.t) * self.dt

    def _normal_cdf(self, x):
        """Standard normal cumulative distribution function"""
        return 0.5 * (1 + math.erf(x / np.sqrt(2)))

    def _get_obs(self):
        # Convert all values to Python floats
        S_t = float(self.path_S[self.t])
        strike = float(self.derivative.strike)
        log_m = float(np.log(S_t / strike))
        tau = float(self._tau())
        vol = float(self.spot.sigma)
        prev_hedge = float(self.prev_hedge)
        cum_pnl = float(self.cum_pnl)
        count = float(self.trade_count)
        
        # Create a numpy array with explicit float32 dtype
        obs = np.array([log_m, tau, vol, prev_hedge, cum_pnl, count], dtype=np.float32)
        
        # Ensure it's the right shape and dtype
        return obs
