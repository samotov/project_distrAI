import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Example: Custom mock environment
from simple_environment import MockEnvironment  # Use your own environment


# Step 1: Create the environment
env = MockEnvironment()
env = DummyVecEnv([lambda: env])  # Vectorize the environment (for compatibility)

# Step 2: Initialize the PPO model
model = PPO("MlpPolicy", env, verbose=1)

# Step 3: Train the model
model.learn(total_timesteps=10000)

# Step 4: Save and load the model
model.save("ppo_mock_env")
loaded_model = PPO.load("ppo_mock_env")

# Step 5: Evaluate the model
state = env.reset()
for _ in range(100):
    action, _ = loaded_model.predict(state)
    state, reward, done, _ = env.step(action)
    env.render()
    if done:
        state = env.reset()

env.close()






