import gymnasium as gym
from gym import spaces
import numpy as np

class MockEnvironment(gym.Env):
    """
    A simple mock environment for PPO testing.
    """

    def __init__(self):
        super(MockEnvironment, self).__init__()

        # Define the action space: Discrete actions (e.g., 0: move left, 1: move right)
        self.action_space = spaces.Discrete(3)  # Three possible actions: [0, 1, 2]

        # Define the observation space: Continuous state representation
        self.observation_space = spaces.Box(
            low=np.array([0, 0]),  # Min values for state features
            high=np.array([1, 1]),  # Max values for state features
            dtype=np.float32
        )

        # Initialize the state
        self.state = np.zeros(2)
        self.step_count = 0
        self.max_steps = 50  # Maximum steps before the environment resets

    def reset(self):
        """
        Reset the environment to the initial state.
        Returns the initial state.
        """
        self.state = np.random.uniform(low=0, high=1, size=(2,))
        self.step_count = 0
        return self.state

    def step(self, action):
        """
        Apply an action to the environment.
        Returns the next state, reward, done, and info.
        """
        # Increment the step count
        self.step_count += 1

        # Update the state based on the action
        if action == 0:  # Move left
            self.state[0] = max(0, self.state[0] - 0.1)
        elif action == 1:  # Move right
            self.state[0] = min(1, self.state[0] + 0.1)
        elif action == 2:  # Stay
            self.state[1] = max(0, self.state[1] - 0.05)

        # Compute the reward (e.g., proximity to a target state)
        target = np.array([0.5, 0.5])
        reward = -np.linalg.norm(self.state - target)  # Negative distance to target

        # Check if the episode is done
        done = self.step_count >= self.max_steps

        # Optional debug info
        info = {"step_count": self.step_count}

        return self.state, reward, done, info

    def render(self, mode="human"):
        """
        Render the environment. For now, just print the state.
        """
        print(f"State: {self.state}")

    def close(self):
        """
        Clean up resources.
        """
        pass


# Example usage
if __name__ == "__main__":
    env = MockEnvironment()

    state = env.reset()
    print("Initial State:", state)

    for _ in range(100):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        print(f"Action: {action}, State: {next_state}, Reward: {reward}, Done: {done}")
        if done:
            break

    env.close()
