import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from sense_env import BasicSynchronousClient
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
import carla
import pygame

# Define symbolic parameters (should match those in sense_env.py)
VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2

class CarlaEnvironment(gym.Env):
    """Custom Environment that wraps CARLA simulation and follows gym interface"""

    def __init__(self):
        super(CarlaEnvironment, self).__init__()

        # Initialize the client
        self.client = BasicSynchronousClient()

        # Initialize pygame
        pygame.init()
        pygame.font.init()

        # Set up the client and world
        self.client.client = carla.Client('127.0.0.1', 2000)
        self.client.client.set_timeout(2.0)
        self.client.world = self.client.client.get_world()

        # Set up the car and sensors
        self.client.setup_car()
        self.client.setup_camera()
        self.client.setup_depth()

        # Set synchronous mode
        self.client.set_synchronous_mode(True)

        # Create display (optional if you want to render)
        self.client.display = pygame.display.set_mode(
            (VIEW_WIDTH, VIEW_HEIGHT),
            pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        self.pygame_clock = pygame.time.Clock()

        # Define action and observation space
        # Actions: [throttle, steer], both in range [-1, 1]
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # Observations: [x, y, speed, heading]
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),
            dtype=np.float32
        )

    def reset(self):
        # Reset the environment and return the initial observation
        self.client.world.tick()
        # Reset the car to a starting position
        spawn_point = self.client.world.get_map().get_spawn_points()[0]
        self.client.car.set_transform(spawn_point)
        self.client.car.apply_control(carla.VehicleControl())

        # Wait for a few ticks to stabilize
        for _ in range(10):
            self.client.world.tick()

        # Get initial observation
        obs = self._get_obs()
        return obs

    def step(self, action):
        # Handle pygame events
        pygame.event.pump()

        # Apply the action to the car
        control = carla.VehicleControl()
        control.throttle = np.clip(action[0], 0.0, 1.0)
        control.steer = np.clip(action[1], -1.0, 1.0)
        control.brake = 0.0
        control.hand_brake = False
        control.reverse = False
        self.client.car.apply_control(control)

        # Advance the simulation
        self.client.world.tick()

        # Capture images (if needed)
        self.client.capture = True
        self.client.capture_depth = True

        # Clock tick
        self.pygame_clock.tick_busy_loop(60)

        # Render images (if needed)
        self.client.render(self.client.display)
        pygame.display.flip()

        # Get the next observation
        obs = self._get_obs()

        # Compute reward
        reward = self._compute_reward()

        # Check if the episode is done
        done = self._is_done()

        # Optionally we can pass additional info
        info = {}

        return obs, reward, done, info

    def render(self, mode='human'):
        # Rendering is handled in the step method
        pass

    def close(self):
        # Clean up resources
        self.client.set_synchronous_mode(False)
        self.client.camera.destroy()
        self.client.depth_camera.destroy()
        self.client.car.destroy()
        pygame.quit()

    def _get_obs(self):
        # Return the current observation
        # For simplicity, let's return the car's position and velocity
        transform = self.client.car.get_transform()
        velocity = self.client.car.get_velocity()
        x = transform.location.x
        y = transform.location.y
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        heading = transform.rotation.yaw

        obs = np.array([x, y, speed, heading], dtype=np.float32)
        return obs

    def _compute_reward(self):
        # Compute the reward for the current state
        # For simplicity, let's define the reward as the speed of the car
        velocity = self.client.car.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        return speed

    def _is_done(self):
        # Check if the episode is over
        # For simplicity, we'll end the episode if the car's speed is below a threshold
        velocity = self.client.car.get_velocity()
        speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
        if speed < 0.1:
            return True
        else:
            return False

def main():
    # Create environment
    env = CarlaEnvironment()

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
    test_env = CarlaEnvironment()  # Initialize without rendering (assuming rendering is handled in sense_env.py)
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
