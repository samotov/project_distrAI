import os
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecFrameStack
from mock_env import Rl_environment
import wandb
import pygame



# Define colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Check GPU availability
device = "cpu"

x = 900
y = 500
os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (x,y)


# Initialize Pygame
pygame.init()
pygame.font.init()
font = pygame.font.SysFont(None, 18)

# Set environment variable to avoid OpenMP issues
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Create the vectorized environment
gym_env = make_vec_env(lambda: Rl_environment(render_mode="human"), n_envs=1)
gym_env = VecFrameStack(gym_env, 3)
gym_env = VecNormalize(gym_env, norm_reward=False)  # Normalize observations for stable learning


# Step 2: Initialize the PPO model
# Instantiate the agent
# Instantiate the PPO agent with the given hyperparameters
RL_model = SAC(
    policy="MlpPolicy",
    env=gym_env,
    buffer_size=2^15,               # Size of the buffer
    learning_starts=256,            # When to start learning
    gamma=0.9,                      # Discount factor
    use_sde=True,                   # State-dependent exploration
    sde_sample_freq=4,              # SDE sampling frequency
    learning_rate=1e-3,             # Learning rate
    verbose=1,                      # Verbosity level
    device=device
)

 # Function to draw a car
def draw_car(screen, x, y, color):
    pygame.draw.rect(screen, color, pygame.Rect(x, y, 50, 30))

def train_model(model, env, total_timesteps, save_interval, save_path="results/ppo_model"):
    """
    Train and evaluate a PPO model, saving progress and stats periodically.

    Parameters:
    - model: The PPO model to train.
    - env: The training environment (should be VecNormalize wrapped if normalization is used).
    - total_time steps: Total number of time steps for training.
    - test_interval: The interval for testing and saving the model.
    - save_path: Base path for saving model checkpoints and environment stats.
    """
    for timestep in range(0, total_timesteps, save_interval):
        print(f"--- Training from timestep {timestep} to {timestep + save_interval} ---")

        # Train the model for the next test_interval steps
        model.learn(total_timesteps=save_interval, progress_bar=True)

        # Save model and environment statistics
        model_save_path = f"{save_path}_{timestep}"
        env_stats_save_path = f"{save_path}_env_stats_{timestep}.pkl"
        model.save(model_save_path)
        env.save(env_stats_save_path)

        if not (timestep == total_timesteps - save_interval):
            evaluate_model(model, env, num_episodes = 2, is_evaluating=False)

    print("Training complete.")
    model.save(f"{save_path}_final")
    return f"{save_path}_final"


def evaluate_model(model, env, num_episodes=5, is_evaluating=True):
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
        # Set up display
        width, height = 800, 160
        x_scale = (width-200)/500
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Car Visualization')

        obs = env.reset()
        reward = 0
        total_reward = 0
        done = False

        while not done:  # Check if all environments are done
            # Render the environment (rendering might not work for all vectorized environments)
            env.render()
            
            # Use deterministic policy during evaluation
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            distance = info[0]["distance"]
            speed = info[0]["speed"]
            lead_speed = info[0]["lead_speed"]
            safe_dist = info[0]["safe_distance"]
            actions = info[0]["actions"]
            total_reward += reward

            lead_car_x = 50
            ego_car_x = lead_car_x + 50 + distance * x_scale
            safe_dist_x = lead_car_x + 50 + safe_dist * x_scale
            safe_dist_l = 0.5 * safe_dist * x_scale


            screen.fill(BLACK)
            draw_car(screen, lead_car_x, 100, BLUE)
            draw_car(screen, ego_car_x, 100, RED)
            pygame.draw.rect(screen, GREEN, pygame.Rect(safe_dist_x , 130, safe_dist_l, 5))
            screen.blit(font.render(f"Distance: {distance:.2f}  Safe distance: {safe_dist:.2f}",True,WHITE),(10,10))
            screen.blit(font.render(f"Current speed: {speed:.2f}   Lead speed: {lead_speed:.2f}",True,WHITE),(10,20))
            screen.blit(font.render(f"Reward step: {reward[0]:.4f}  Total reward: {total_reward[0]:.4f}",True,WHITE),(10,30))
            screen.blit(font.render(f"Actions: brake {actions[0]:.4f}, throttle: {actions[1]:.4f}",True,WHITE),(10,40))
            
            # Update the screen
            pygame.display.flip()

            if is_evaluating:
                wandb.log({
                    f"test/reward_{episode}": reward,
                    f"test/total_reward_{episode}": total_reward,
                    f"test/distance_{episode}": info[0]["distance"],
                    f"test/speed_{episode}":  info[0]["speed"]
                    })
        
        total_rewards.append(total_reward)
        print(f"Episode {episode + 1}/{num_episodes} reward: {total_reward.item():.2f}, stopped because {info[0]["stop_condition"]}" + " " * 10)
        

    print(f"Average reward after training: {(sum(total_rewards) / num_episodes)[0]}")
    
# =======================================================================================

if __name__ == "__main__":
    print("start")
    # Define the total number of steps
    total_timesteps = 100000

    # Test the model after every 1000 timesteps (periodically)
    test_interval = total_timesteps // 100

    # Train and evaluate
    try:
        train_model(RL_model, gym_env, total_timesteps, test_interval)
        # RL_model = PPO.load("./results/ppo_model_final")
        evaluate_model(RL_model, gym_env, 5)
    finally:
        # Close the environment to release resources
        gym_env.close()
        pygame.quit()
        
