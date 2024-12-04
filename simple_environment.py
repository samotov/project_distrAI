import numpy as np
import random
import gymnasium as gym
from gym.spaces import Box
from reward_calculation import calculate_reward

class SpeedAdjustmentEnv(gym.Env):
    def __init__(self, render_mode="human"):
        super(SpeedAdjustmentEnv, self).__init__()

        # Lower -and higher bounds for the speed
        self.LOW_BOUND = 0.0
        self.HIGH_BOUND = 100.0

        # self.STEERING_ANGLE  =
        # self.VELOCITY
        #self.STOPPING_DISTANCE =


        # Action space: Continuous space representing the car's speed
        self.action_space = gym.spaces.Box(low=self.LOW_BOUND, high=self.HIGH_BOUND, shape=(1,), dtype=np.float32)

        # Observation space: Example of a state with target speed and current speed
        self.observation_space = gym.spaces.Box(low=0.0, high=100.0, shape=(2,), dtype=np.float32)

        # Initial state
        self.current_speed = 0.0
        self.target_speed = 30.0  # Example target speed

        self.max_steps = 50  # Maximum steps per episode
        self.step_count = 0

        # Store the render mode
        self.render_mode = render_mode

    def reset(self ,*, seed=None, return_info=False, options=None):
        # Reset the environment to its initial state
        self.current_speed = np.random.uniform(self.LOW_BOUND, self.HIGH_BOUND)
        self.step_count = 0
        return np.array([self.current_speed, self.target_speed], dtype=np.float32)

    def step(self, action):
        # Clip the action to ensure it's within the valid range
        action = np.clip(action, self.LOW_BOUND, self.HIGH_BOUND)

        # Apply the action to adjust the car's speed
        self.current_speed += (action[0] - self.current_speed) * 0.1  # Smooth adjustment

        # Calculate the reward based on the closeness to the target speed
        speed_diff = abs(self.current_speed - self.target_speed)

        # Calculate the total reward for this action
        reward = calculate_reward(speed_diff)

        # Define the done condition
        self.step_count += 1

        terminated = self.step_count >= self.max_steps  # Episode termination condition
        truncated = False  # No time limit for truncation in this environment

        # Observation includes current speed and target speed
        obs = np.array([self.current_speed, self.target_speed], dtype=np.float32)

        # Returning 5 values: observation, reward, terminated, truncated, and info
        info = {}  # Optional additional info

        return obs, reward, terminated, truncated, info


    def render(self, mode="human"):
        if self.render_mode == "human":
            print(
                f"Step: {self.step_count}, Current Speed: {self.current_speed:.2f}, Target Speed: {self.target_speed:.2f}")

    def seed(self, seed=None):
        # Set the seed for random number generation
        self.np_random = np.random.default_rng(seed)
        return [seed]

    def close(self):
        pass


