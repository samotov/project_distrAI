from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
import gym
import numpy as np
import pickle

class ACCController():
    def __init__(self, model_path: str = None, env_path: str = None):
    
        self.model = PPO.load(model_path, device="cpu")

        # Extract normalization parameters
        self.normalize = self._extract_normalization_model(env_path)

        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))
        

    def predict(self, obs: np.ndarray):
        """
        Predict the actions to take based on the observations
        """
        # Normalize observations
        norm_obs: np.ndarray = self.normalize(obs) 

        # Use trained model to predict next actions
        actions: np.ndarray = self.model.predict(norm_obs)
        throttle, brake = np.clip(action, self.action_space.low, self.action_space.high)

        return np.float(throttle), np.float(brake)
        
    def _extract_normalization_model(self, env_path: str):
     # Load data from environment
        with open(env_path, 'rb') as file:
            norm_env = pickle.load(file)
            normalize = self._create_normalization_func(norm_env)

        del norm_env
        return normalize

    @staticmethod
    def _create_normalization_func(norm_env: VecNormalize):
        """
        Create normalization function based on parameters from trained VecNorm environment.
        """
        # Create normalization function as defined in the VecNormalize module
        def normalize(obs: np.ndarray) -> np.ndarray:
            # Mean and variance of running mean square
            obs_mean = norm_env.obs_rms.mean
            obs_var = norm_env.obs_rms.var

            # Epsilon to avoid division by zero
            eps = norm_env.epsilon

            # Clip values of observation
            clip_obs = norm_env.clip_obs

            # Normalized observations
            return np.clip((obs - obs_mean) / np.sqrt(obs_var + eps), -clip_obs, clip_obs)
    
        # Return the function
        return normalize
