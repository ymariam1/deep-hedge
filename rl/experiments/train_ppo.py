#!/usr/bin/env python
# rl/experiments/train_ppo.py
"""
Train a PPO agent to learn no-transaction-band widths that minimise tail risk.

Usage
-----
python train_ppo.py                           # default 2 M steps
python train_ppo.py --timesteps 5_000_000     # longer run
python train_ppo.py --seed 123 --tag test     # reproducible run labelled 'test'
"""

import argparse, datetime, pathlib, json, os
import sys
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from gymnasium import Wrapper

# Add the project root to the Python path so we can import from src
sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from src.market.underlying.gbm_stock import BrownianStock
from src.market.derivative.european_option import EuropeanOption
# Import HedgingEnv directly to avoid package issues
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from env.hedging_env import HedgingEnv


# Add a wrapper that ensures observations are compatible with PyTorch
class DirectObsWrapper(Wrapper):
    """Handles observation conversion to PyTorch compatible format."""
    
    def __init__(self, env):
        super().__init__(env)
        # Force observations to be handled as Torch tensors directly
        self.observation_space = env.observation_space
        
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


def make_env(seed: int, steps: int, shortfall: float, cost: float):
    """Factory for creating environment."""
    def _init():
        spot = BrownianStock(sigma=0.2, mu=0.0, cost=cost)
        deriv = EuropeanOption(spot, maturity=1.0, strike=1.0)
        env = HedgingEnv(deriv, spot_model=spot,
                         steps=steps,
                         cost=cost,
                         shortfall_thr=shortfall)
        # Set the seed first
        env.reset(seed=seed) 
        # Wrap with our direct observation wrapper
        return DirectObsWrapper(env)
    return _init


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--timesteps", type=int, default=2_000_000)
    pa.add_argument("--steps", type=int, default=250, help="hedge rebalancing steps")
    pa.add_argument("--shortfall", type=float, default=-0.10,
                    help="CVaR hinge threshold (in $)")
    pa.add_argument("--cost", type=float, default=1e-4)
    pa.add_argument("--lr", type=float, default=3e-4)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--tag", type=str, default="run")
    args = pa.parse_args()

    # ---------- output directory ----------
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = pathlib.Path("rl/results/ppo_models") / f"{args.tag}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Create vectorized env using SB3's utility function ------------
    vec_env = make_vec_env(
        env_id=lambda: make_env(args.seed, args.steps, args.shortfall, args.cost)(),
        n_envs=1,
        seed=args.seed,
        vec_env_cls=DummyVecEnv
    )

    # Test a sample observation to verify it works
    print("Testing environment...")
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
    
    # ---------- PPO hyper-parameters -------
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = PPO("MlpPolicy",
                vec_env,
                learning_rate=args.lr,
                n_steps=2048,
                batch_size=256,
                gamma=1.0,               # episodic reward (no discount)
                gae_lambda=1.0,
                clip_range=0.2,
                ent_coef=0.0,
                policy_kwargs=policy_kwargs,
                tensorboard_log=str(out_dir / "tb"),
                seed=args.seed,
                verbose=1)

    # --------- checkpoint callback ---------
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000,
        save_path=str(out_dir / "ckpt"),
        name_prefix="ppo_band"
    )

    # -------------- train ------------------
    try:
        model.learn(total_timesteps=args.timesteps,
                    callback=checkpoint_cb,
                    progress_bar=True)

        # -------------- save -------------------
        model_path = out_dir / "ppo_band_final.zip"
        model.save(model_path)

        # Save meta-data for reproducibility
        meta = vars(args) | {"model_path": str(model_path)}
        with open(out_dir / "run_config.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nFinished training • model saved to {model_path}")
    
    except Exception as train_error:
        print(f"Training error: {train_error}")
        
        # If training fails, we'll print the PyTorch and NumPy versions
        print("\nEnvironment details:")
        print(f"PyTorch version: {torch.__version__}")
        print(f"NumPy version: {np.__version__}")
        print(f"Python version: {sys.version}")
        
        # Suggest potential solutions
        print("\nPotential solutions:")
        print("1. Try downgrading PyTorch: pip install torch==2.0.1")
        print("2. Try upgrading NumPy: pip install numpy==1.24.3")
        print("3. Use a different conda environment with compatible versions")

if __name__ == "__main__":
    main()
