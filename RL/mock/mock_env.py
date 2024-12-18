import gymnasium as gym
import numpy as np
import wandb



class Rl_environment(gym.Env):
    def __init__(self, render_mode="human", config={}):  # Pass the render_mode during initialization
        super(Rl_environment, self).__init__()
        
        # Init wandb
        wandb.init(project="ACC_agent", config=config)

        self.render_mode = render_mode  # Store the render_mode
        self.maximum_distance = 500

        # Define observation space: distance to lead vehicle
        self.observation_space = gym.spaces.Box(low=np.array([0., 0., 1., 1.],  dtype=np.float32), high= np.array([self.maximum_distance, 22.77, 1, 1],  dtype=np.float32))

        # Define action space: continuous throttle and brake
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

        # Environment parameters
        self.SENSOR_TICK = 0.03  # Sensor tick time in seconds
        self.target_speed_lead = 13.88  # Lead vehicle target speed (m/s) = 50 km/h
        self.safe_distance = 20  # Minimum safe following distance (meters)
        self.collision_distance = 1
        self.green_light_duration = 40  # Duration of green light (seconds)
        self.red_light_duration = 10  # Duration of red light (seconds)
        self.lead_acceleration = 2.0  # Lead vehicle acceleration rate (m/s^2)
        self.lead_deceleration = -3.0  # Lead vehicle deceleration rate (m/s^2)
        
        # Car size
        self.car_width, self.car_height = 50, 30

        # Initialize positions for the cars
        self.car1_x = 720
        self.car2_x = 0

        # Initialize state
        self.distance = None
        self.previous_distance = None
        self.lead_vehicle_speed = None
        self.traffic_light_state = None  # 0 for green, 1 for red
        self.time_in_light_state = None
        self.ego_speed = None
        self.previous_ego_speed = None  # To track speed change
        self.time_without_acceleration = 0  # Time without acceleration
        self.time = 0
        self.reset()

    def _get_obs(self):
        return np.array([self.distance, self.ego_speed, 1, 1], dtype=np.float32)

    def _get_action(self, action):
        if action > 0.5:
            throttle =  2 * (action.item() - 0.5)
            brake = 0
        elif action < 0.5:
            brake = 2 * action.item()
            throttle = 0
        else:
            throttle = 0
            brake = 0

        return throttle, brake

    def reset(self, seed=None):
        # Reset state
        self.ego_speed = np.random.uniform(2.77, 18)  # Ego vehicle initial speed: 10-50 km/h (expressed in m/s)
        self.previous_ego_speed = self.ego_speed
        self.lead_vehicle_speed = self.target_speed_lead
        self.distance = np.random.uniform(50, self.maximum_distance/2)  # Random initial distance to lead vehicle
        self.previous_distance = self.distance
        self.traffic_light_state = 0  # Start with green light
        self.time_in_light_state = 0
        self.time_without_acceleration = 0  # Reset the time without acceleration
        self.time = 0

        # Return observation and info
        observation = self._get_obs()
        info = {}  # Optionally include additional information in the info dictionary

        return observation, info

    def step(self, action):
        info = {}

        self.previous_distance = self.distance
        self.previous_ego_speed = self.ego_speed

        self.time += self.SENSOR_TICK

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Ego vehicle dynamics
        throttle, brake = self._get_action(action)
        self.ego_speed += (throttle * 2 - brake * 3) * self.SENSOR_TICK
        self.ego_speed = np.clip(self.ego_speed, 0, 27.77)  # Max 100 km/h
        self.safe_distance = 100

        # Lead vehicle dynamics
        # if self.traffic_light_state == 1:  # Red light
        #     self.lead_vehicle_speed += self.lead_deceleration * self.SENSOR_TICK
        #     self.lead_vehicle_speed = max(self.lead_vehicle_speed, 0)  # Stop at 0 speed
        # else:  # Green light
        #     self.lead_vehicle_speed += self.lead_acceleration * self.SENSOR_TICK
        #     self.lead_vehicle_speed = min(self.lead_vehicle_speed, self.target_speed_lead)  # Max to target speed

        # Update distance
        self.distance -= (self.ego_speed - self.target_speed_lead) * self.SENSOR_TICK
        self.distance = np.clip(self.distance, 0, self.maximum_distance)

        # Traffic light logic
        self.time_in_light_state += self.SENSOR_TICK
        # if self.traffic_light_state == 0 and self.time_in_light_state >= self.green_light_duration:
        #     self.traffic_light_state = 1  # Switch to red
        #     self.time_in_light_state = 0
        # elif self.traffic_light_state == 1 and self.time_in_light_state >= self.red_light_duration:
        #     self.traffic_light_state = 0  # Switch to green
        #     self.time_in_light_state = 0

        # Check if ego vehicle is stationary
        if self.ego_speed < 0.1 and self.distance > self.safe_distance:  # Consider as stationary if speed is near zero
            self.time_without_acceleration += self.SENSOR_TICK
        else:
            self.time_without_acceleration = 0

        speed_penalty = abs(self.ego_speed - self.lead_vehicle_speed)/(27.77 - self.lead_vehicle_speed)
        control_penalty = throttle + brake
        distance_penalty = abs(self.distance - self.safe_distance)/(self.maximum_distance - self.safe_distance)
        reward = (1 - distance_penalty)

        # if abs(self.ego_speed - self.lead_vehicle_speed) < 0.2 * self.lead_vehicle_speed:
        #     reward += 0.5
        
        terminated = False
        
        # Punish for collision
        if self.distance <= self.collision_distance:  # Collision
            reward = -2
            terminated = True
            info["stop_condition"] = "Collision"

        # # Punish for inactiviy
        elif self.time_without_acceleration >= 8:  # Terminate if vehicle doesn't move for 8 seconds
            reward = 0
            terminated = True
            info["stop_condition"] = "Inactivity"

        elif self.distance < self.safe_distance * 0.1:
            reward = 0

        elif self.distance < self.safe_distance:
            reward = (1 - 3 * distance_penalty)
            # terminated = True
            # info["stop_condition"] = "Too close"

        elif self.distance < self.safe_distance * 1.5:
            reward = 1
            
        truncated = False
        if self.time > self.maximum_distance:
            truncated = True
            info["stop_condition"] = f"Truncated at {round(self.time)} seconds"

        info["distance"] = self.distance
        info["speed"] = self.ego_speed
        info["safe_distance"] = self.safe_distance
        info["lead_speed"] = self.lead_vehicle_speed
        info["actions"] = (brake, throttle)

        # Return observation, reward, terminated, truncated, and info
        return self._get_obs(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(
                f"Distance to Lead: {self.distance:.2f}, Ego Speed: {self.ego_speed:.2f}, "
                f"Lead Speed: {self.lead_vehicle_speed:.2f}", end = "\r")
            
           

   
