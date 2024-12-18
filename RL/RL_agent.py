import os
import torch

import spawn
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from CarlaEnv import CarlaEnv

# Check GPU availability
device = "cpu"


# Set environment variable to avoid OpenMP issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create the vectorized environment
gym_env = make_vec_env(lambda: CarlaEnv(), n_envs=1)
gym_env = VecNormalize(gym_env, norm_reward=False)  # Normalize observations for stable learning

# Step 2: Initialize the PPO model
# Instantiate the agent
# Instantiate the PPO agent with the given hyperparameters
RL_model = SAC(
    policy="MlpPolicy",
    env=gym_env,
    gamma=0.9,                     # Discount factor
    verbose=1                      # Verbosity level
)


def train_and_evaluate(model, env, total_timesteps, test_interval, save_path="results/ppo_model"):
    """
    Train and evaluate a PPO model, saving progress and stats periodically.

    Parameters:
    - model: The PPO model to train.
    - env: The training environment (should be VecNormalize wrapped if normalization is used).
    - total_time steps: Total number of time steps for training.
    - test_interval: The interval for testing and saving the model.
    - save_path: Base path for saving model checkpoints and environment stats.
    """
    for timestep in range(0, total_timesteps, test_interval):
        print(f"\n--- Training from timestep {timestep} to {timestep + test_interval} ---")

        # Train the model for the next test_interval steps
        model.learn(total_timesteps = test_interval, 
                    reset_num_timesteps = False,
                    progress_bar=True)
        print(f"Training completed for {test_interval} timesteps.")

        # Save model and environment statistics
        model_save_path = f"{save_path}_{timestep}.zip"
        env_stats_save_path = f"{save_path}_env_stats_{timestep}.pkl"
        model.save(model_save_path)
        env.save(env_stats_save_path)
        print(f"Model and environment stats saved at timestep {timestep}.")

        # Evaluate the model
        total_reward = evaluate_model(env, model)
        print(f"After {timestep} time steps, average reward: {total_reward.item():.2f}")

    print("\nTraining and evaluation complete.")


def evaluate_model(env, model, num_episodes=1):
    """
    Evaluate a PPO model over multiple episodes.

    Parameters:
    - env: The environment for evaluation.
    - model: The trained PPO model.
    - num_episodes: Number of episodes to evaluate over.

    Returns:
    - Average reward across all evaluated episodes.
    """
    total_rewards = []

    for episode in range(num_episodes):
        obs = env.reset()
        reward = 0
        total_reward = 0
        done = False

        while not done:
            # Use deterministic policy during evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} reward: {total_reward.item():.2f}")

    avg_reward = sum(total_rewards) / num_episodes
    return avg_reward


# =======================================================================================

# Define the total number of steps
total_timesteps = 10240

# Test the model after every 1000 timesteps (periodically)
test_interval = 1024

# Train and evaluate
try:
    
    train_and_evaluate(RL_model, gym_env, total_timesteps, test_interval)
finally:
    # Close the environment to release resources
    gym_env.close()

