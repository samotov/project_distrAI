import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from simple_environment import SpeedAdjustmentEnv

import torch


print(torch.cuda.is_available())  # Should return True if GPU is available


# Set environment variable to avoid OpenMP issues
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create the vectorized environment
env = make_vec_env(lambda: SpeedAdjustmentEnv(render_mode="human"), n_envs=1)  # Use lambda to initialize the env
env = VecNormalize(env, norm_reward=False)

# Step 2: Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Step 3: Train the model
model.learn(total_timesteps=10000)

# Save the model after training
model.save("ppo_speed_adjustment")

# Optionally, you can load and test the trained model after saving
# loaded_model = PPO.load("ppo_speed_adjustment")




