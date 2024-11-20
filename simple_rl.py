import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from simple_environment import SimpleEnvironment
from stable_baselines3.common.vec_env import DummyVecEnv

def main():
    # Create environment
    env = SimpleEnvironment()

    # Optionally wrap the environment to limit the length of each episode
    env = gym.wrappers.TimeLimit(env, max_episode_steps=500)

    # Wrap the environment in a DummyVecEnv (required for Stable-Baselines3 PPO)
    env = DummyVecEnv([lambda: env])

    # Create a PPO agent using an MLP policy
    model = PPO('MlpPolicy', env, verbose=1)

    # Set up a callback to save the model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,  # Save the model every 10,000 timesteps
        save_path='./logs/',  # Directory to save the model
        name_prefix='ppo_carla_model'  # Prefix for the saved model files
    )

    # Train the agent
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

    # Save the final trained model
    model.save("ppo_carla_final")

    # Test the trained agent using an unvectorized environment
    test_env = SimpleEnvironment(render_mode='human')  # Initialize with rendering
    obs = test_env.reset()
    for _ in range(1000):  # Run for 1000 steps
        action, _ = model.predict(obs, deterministic=True)  # Get the agent's action
        obs, reward, done, info = test_env.step(action)  # Take the action in the environment
        test_env.render()  # Render the environment
        if done:  # Reset the environment if an episode ends
            obs = test_env.reset()
    test_env.close()  # Close the testing environment

if __name__ == '__main__':
    main()
