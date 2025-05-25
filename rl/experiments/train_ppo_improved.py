#!/usr/bin/env python
# rl/experiments/train_ppo_improved.py
"""
Train a PPO agent with improved reward function that prevents fat tails.

The key improvements:
1. Cap reward at 0 when P&L > shortfall threshold (prevents gambling)
2. Add variance penalty to discourage high-variance strategies
3. Maintain trade cost penalties for efficiency

Usage
-----
python train_ppo_improved.py                           # default 2 M steps
python train_ppo_improved.py --timesteps 5_000_000     # longer run
python train_ppo_improved.py --seed 123 --tag improved # reproducible run
"""

import argparse, datetime, pathlib, json, os
import sys
import numpy as np
import torch

# Add the project root to the Python path so we can import from src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.env_util import make_vec_env
    from gymnasium import Wrapper
    STABLE_BASELINES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: stable-baselines3 not available: {e}")
    print("You can install it with: pip install stable-baselines3[extra]")
    STABLE_BASELINES_AVAILABLE = False

from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
# Import HedgingEnv directly to avoid package issues
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.hedging_env import HedgingEnv


# Add a wrapper that ensures observations are compatible with PyTorch
class DirectObsWrapper:
    """Handles observation conversion to PyTorch compatible format."""
    
    def __init__(self, env):
        self.env = env
        # Copy important attributes from the wrapped environment
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        # Use explicit copy to avoid PyTorch/NumPy compatibility issues
        # This is slower but guaranteed to work
        obs_copy = np.array(obs, dtype=np.float32, copy=True)
        # Ensure it's contiguous and properly formatted
        if not obs_copy.flags.c_contiguous:
            obs_copy = np.ascontiguousarray(obs_copy)
        return obs_copy
    
    def close(self):
        """Close the environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def seed(self, seed=None):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
        return []
    
    def render(self, mode='human'):
        """Render the environment."""
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
    
    def __getattr__(self, name):
        """Delegate any other attributes to the wrapped environment."""
        return getattr(self.env, name)


# Proper Gymnasium wrapper for stable-baselines3 compatibility
if STABLE_BASELINES_AVAILABLE:
    from gymnasium import Wrapper
    
    class ProperDirectObsWrapper(Wrapper):
        """Proper Gymnasium wrapper for stable-baselines3 compatibility."""
        
        def __init__(self, env):
            super().__init__(env)
        
        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self._process_obs(obs), info
        
        def step(self, action):
            obs, reward, terminated, truncated, info = self.env.step(action)
            return self._process_obs(obs), reward, terminated, truncated, info
        
        def _process_obs(self, obs):
            # Use explicit copy to avoid PyTorch/NumPy compatibility issues
            obs_copy = np.array(obs, dtype=np.float32, copy=True)
            # Ensure it's contiguous and properly formatted
            if not obs_copy.flags.c_contiguous:
                obs_copy = np.ascontiguousarray(obs_copy)
            return obs_copy


def make_env(seed: int, steps: int, shortfall: float, cost: float):
    """Factory for creating environment with improved reward function."""
    def _init():
        spot = BrownianStock(sigma=0.2, mu=0.0, cost=cost)
        deriv = EuropeanOption(spot, maturity=1.0, strike=1.0)
        env = HedgingEnv(deriv, spot_model=spot,
                         steps=steps,
                         cost=cost,
                         shortfall_thr=shortfall)
        # Set the seed first
        env.reset(seed=seed) 
        
        # Use appropriate wrapper based on availability
        if STABLE_BASELINES_AVAILABLE:
            return ProperDirectObsWrapper(env)
        else:
            return DirectObsWrapper(env)
    return _init


def main():
    if not STABLE_BASELINES_AVAILABLE:
        print("ERROR: stable-baselines3 is required for PPO training.")
        print("Please install it with one of the following commands:")
        print("  pip install stable-baselines3[extra]")
        print("  conda install -c conda-forge stable-baselines3")
        print("\nAlternatively, fix your environment dependencies:")
        print("  - Check numpy/pandas compatibility")
        print("  - Consider creating a fresh conda environment")
        sys.exit(1)

    pa = argparse.ArgumentParser()
    pa.add_argument("--timesteps", type=int, default=2_000_000)
    pa.add_argument("--steps", type=int, default=250, help="hedge rebalancing steps")
    pa.add_argument("--shortfall", type=float, default=-0.10,
                    help="CVaR hinge threshold (in $)")
    pa.add_argument("--cost", type=float, default=1e-4)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--tag", type=str, default="improved")
    args = pa.parse_args()

    # ---------- output directory ----------
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path("rl/results/ppo_models") / f"{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Training PPO with improved reward function (no fat tails)")
    print(f"Key improvements:")
    print(f"  - Reward capped at 0 when P&L > {args.shortfall}")
    print(f"  - Variance penalty to discourage high-variance strategies")
    print(f"  - Trade cost penalties maintained")
    print(f"Output directory: {out_dir}")
    print(f"Using wrapper: {'ProperDirectObsWrapper' if STABLE_BASELINES_AVAILABLE else 'DirectObsWrapper'}")

    # ---------- Create vectorized env using SB3's utility function ------------
    try:
        print("\nCreating vectorized environment...")
        vec_env = make_vec_env(
            env_id=lambda: make_env(args.seed, args.steps, args.shortfall, args.cost)(),
            n_envs=1,
            seed=args.seed,
            vec_env_cls=DummyVecEnv
        )
        print("✓ Vectorized environment created successfully")
    except Exception as e:
        print(f"✗ Error creating vectorized environment: {e}")
        print("Attempting single environment test...")
        
        # Test single environment
        try:
            env = make_env(args.seed, args.steps, args.shortfall, args.cost)()
            obs, info = env.reset()
            print(f"✓ Single environment test successful: obs shape {obs.shape}")
            
            # Try a single step
            action = np.array([0.1, 0.1], dtype=np.float32)
            obs, reward, done, truncated, info = env.step(action)
            print(f"✓ Single step test successful: reward={reward:.6f}")
            env.close()
            
            # Create a simple single-env wrapper for training
            print("Creating simple single environment for training...")
            single_env = make_env(args.seed, args.steps, args.shortfall, args.cost)()
            
            # Wrap in DummyVecEnv manually
            vec_env = DummyVecEnv([lambda: single_env])
            print("✓ Manual vectorized environment created")
            
        except Exception as single_env_error:
            print(f"✗ Single environment test failed: {single_env_error}")
            print("Environment setup failed completely. Exiting...")
            return

    # Test a sample observation to verify it works
    print("\nTesting environment...")
    obs = vec_env.reset()
    print(f"Observation: shape={obs.shape}, dtype={obs.dtype}")
    print(f"Sample: {obs[0]}")

    # Try explicit tensor conversion to check if it works
    try:
        print("Testing tensor conversion...")
        # Use the same explicit copy approach as our wrapper
        obs_copy = np.array(obs, dtype=np.float32, copy=True)
        if not obs_copy.flags.c_contiguous:
            obs_copy = np.ascontiguousarray(obs_copy)
        tensor_obs = torch.tensor(obs_copy, dtype=torch.float32)
        print("✓ Tensor conversion successful")
    except Exception as e:
        print(f"✗ Tensor conversion failed: {e}")
        print("This suggests a deeper PyTorch/NumPy compatibility issue")
        return
    
    # ---------- PPO hyper-parameters (tuned for stable hedging) -------
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    
    try:
        model = PPO("MlpPolicy",
                    vec_env,
                    learning_rate=args.lr,
                    n_steps=2048,
                    batch_size=256,
                    gamma=1.0,               # episodic reward (no discount)
                    gae_lambda=1.0,
                    clip_range=0.2,
                    ent_coef=0.01,           # Slightly higher entropy for exploration
                    policy_kwargs=policy_kwargs,
                    tensorboard_log=str(out_dir / "tb"),
                    seed=args.seed,
                    verbose=1)
    except Exception as e:
        print(f"Error creating PPO model: {e}")
        print("This might be due to environment or dependency issues.")
        return

    # --------- checkpoint callback ---------
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(out_dir / "ckpt"),
        name_prefix="ppo_band_improved"
    )

    # -------------- train ------------------
    try:
        print(f"\nStarting training for {args.timesteps:,} timesteps...")
        model.learn(total_timesteps=args.timesteps,
                    callback=checkpoint_cb,
                    progress_bar=True)

        # -------------- save -------------------
        model_path = out_dir / "ppo_band_improved_final.zip"
        model.save(model_path)

        # Save meta-data for reproducibility
        meta = vars(args) | {
            "model_path": str(model_path),
            "improvements": [
                "Reward capped at 0 when P&L > shortfall threshold",
                "Variance penalty to discourage fat tails", 
                "Trade cost penalties maintained"
            ]
        }
        with open(out_dir / "run_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\n✓ Training completed successfully!")
        print(f"✓ Model saved to {model_path}")
        print(f"✓ Configuration saved to {out_dir / 'run_config.json'}")
        print(f"\nTo evaluate the improved model:")
        print(f"python rl/experiments/evaluate.py")
        print(f"\nTo compare with baselines:")
        print(f"python scripts/compare_baselines.py --ppo-model {model_path} --save")
    
    except Exception as train_error:
        print(f"✗ Training error: {train_error}")
        
        # If training fails, we'll print the PyTorch and NumPy versions
        print("\nEnvironment details:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Python version: {sys.version}")
        
        # Suggest potential solutions
        print("\nPotential solutions:")
        print("1. Create a fresh conda environment:")
        print("   conda create -n deep_hedge python=3.9")
        print("   conda activate deep_hedge")
        print("   pip install torch torchvision stable-baselines3[extra]")
        print("2. Fix dependency conflicts:")
        print("   pip install --upgrade numpy pandas")
        print("3. Use CPU-only versions if GPU causing issues:")
        print("   pip install torch --index-url https://download.pytorch.org/whl/cpu")

if __name__ == "__main__":
    main() 