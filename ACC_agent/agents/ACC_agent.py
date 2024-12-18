import carla

from .basic_agent import BasicAgent
from controllers import ACCController
from sensors import DepthSensor, RGBSensor, ObjectDetector
from queue import Queue
from PIL import Image
from collections import deque
import pygame
import numpy as np
import os
import random


class ACCAgent(BasicAgent):
    """
    ACC Agent is a AI-based agent, drawing from the Basic Agent module.
    The ACC Agent uses steering from the Basic Agent, but controls its throttle and brake using an RL agent.
    For this, the ACC agent detects other vehicles, using computer vision, and estimates its distance. 
    If the vehicle is within its motion tube, it will consider the distance between itself and the vehicle and adapt its speed. 
    """

    # Define symbolic parameters
    VIEW_WIDTH = 1920//2
    VIEW_HEIGHT = 1080//2
    VIEW_FOV = 90

    MIN_RANGE = 0.5             # Minimum range for depth sensor in meters
    MAX_RANGE = 50.0            # Maximum range for depth sensor in meters
    SENSOR_TICK = 0.1           # Sensor tick time in seconds

    def __init__(self, vehicle, target_speed=20, opt_dict={}, map_inst=None, grp_inst=None):
        """
        Initialization the agent parameters, the local and the global planner.

            :param vehicle: actor to apply to agent logic onto
            :param target_speed: speed (in Km/h) at which the vehicle will move
            :param opt_dict: dictionary in case some of its parameters want to be changed.
                This also applies to parameters related to the LocalPlanner.
            :param map_inst: carla.Map instance to avoid the expensive call of getting it.
            :param grp_inst: GlobalRoutePlanner instance to avoid the expensive call of getting it.
        """
        super().__init__(vehicle, target_speed, opt_dict=opt_dict, map_inst=map_inst, grp_inst=grp_inst)

        self._current_speed = vehicle.get_velocity().length()   # [m/s]

        # AI components
        self._object_detector = ObjectDetector()
        self._ACC_controller = self._setup_ACC_controller("models/ACC_controller_model/ppo_constant_target_speed.zip", "models/ACC_controller_model/env_constant_target_speed.pkl")
        
        # Sensors
        self._collision_sensor = None  
        self._depth_sensor = None
        self._rgb_sensor = None

        # Setup sensors
        self._setup_RGB_camera()
        self._setup_depth_camera()
        self._set_collision_sensor()

        # Everything for storing the obtained images
        self.RGB_image = None
        self.depth_image = None
        self.depth_logarithmic = None
        self._image_dictionary = {"rgb_image": None, "bgr_depth":None, "log_depth": None} # Object that will hold all the information from the sensors

        # Actors (i.e. vehicles, walkers, lights) that have been spotted
        self.spotted_actors = None

        # View of the RGB camera
        self.view_width = 1920//2
        self.view_height = 1080//2

        # Initialize a deque to store the last 10 steering angles
        self.steering_angle_history = deque(maxlen=10)
        self.left_motion_tube_points = []
        self.right_motion_tube_points = []
        
    # ==============================================================================================
    # Agent setup
    # ==============================================================================================

    def _set_object_detector(self):
        """Instantiate the object detector"""
        self._object_detector = ObjectDetector()

    def _extract_RGB_image(self, image):
        """Extract RGB image from the sensor and save it in the dictionary"""
        rgb_image = self._rgb_sensor.capture_image(image)
        self._image_dictionary["rgb_image"] = rgb_image

    def _extract_depth_image(self, image):
        """Extract RGB image from the sensor and save it in the dictionary"""
        bgr_depth, grayscale_depth = self._depth_sensor.capture_image(image)
        self._image_dictionary["bgr_depth"] = bgr_depth
        self._image_dictionary["log_depth"] = grayscale_depth

    def _setup_RGB_camera(self):
        """Instantiate the RGB camera"""
        # Create RGB sensor class that will handle everything concerning rgb images
        self._rgb_sensor = RGBSensor()

        # Get updated RGB settigns
        rgb_bp = self._world.get_blueprint_library().find("sensor.camera.rgb")
        rgb_camera_bp, calibration = self._rgb_sensor.setup_rgb_camera(rgb_bp, self._vehicle)
        
        # Create the RGB camera
        rgb_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.rgb_camera = self._world.spawn_actor(rgb_camera_bp, rgb_transform, attach_to=self._vehicle)
        self.rgb_camera.calibration = calibration

        # Make the camera register images
        self.rgb_camera.listen(lambda rgb_image: self._extract_RGB_image(rgb_image))
    
    def _setup_depth_camera(self):       
        """
        Returns depth blueprint with specified symbolic attributes.
        """
        # Create depth sensor class that will handle everything concerning depth images
        self._depth_sensor = DepthSensor()

        # Get updated depth settigns
        depth_bp = self._world.get_blueprint_library().find("sensor.camera.depth")
        depth_bp, calibration = self._depth_sensor.setup_depth_camera(depth_bp, self._vehicle)
        
        # Create the depth camera
        depth_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.depth_camera = self._world.spawn_actor(depth_bp, depth_transform, attach_to=self._vehicle)
        self.depth_camera.calibration = calibration

        # Make the camera register images
        self.depth_camera.listen(lambda depth_image: self._extract_depth_image(depth_image))
    
    def _setup_ACC_controller(self, model_path: str, env_path: str):
        """Create the ACC controller which will decide the throttle and brake"""
        self._ACC_controller = ACCController(model_path, env_path)

    def _set_collision_sensor(self):
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(lambda event: self.stop_constant_velocity())

    
    # ==============================================================================================
    # Calculate motion tubes
    # ==============================================================================================

    def generate_curve_points(self, start, control, end, num_points=50):
        """
        Generate points along a quadratic Bezier curve.
        :param start: Tuple (x, y) for the start point
        :param control: Tuple (x, y) for the control point
        :param end: Tuple (x, y) for the end point
        :param num_points: Number of points to generate along the curve
        :return: List of (x, y) points
        """
        t_values = np.linspace(0, 1, num_points)
        points = [
            (
                (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * control[0] + t ** 2 * end[0],
                (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * control[1] + t ** 2 * end[1],
            )
            for t in t_values
        ]
        return points
                    
        
    def _calculate_motion_tube(self):
        """Calculate the different points needed to create the motion tubes"""
        # Get the current steering angle
        current_steering_angle = self._vehicle.get_control().steer  # Value between -1.0 (left) and 1.0 (right)

        # Add the current steering angle to the history
        self.steering_angle_history.append(current_steering_angle)

        # Compute the average steering angle over the last 20 frames
        if len(self.steering_angle_history) > 20:
            self.steering_angle_history.pop(0)
        avg_steering_angle = sum(self.steering_angle_history) / len(self.steering_angle_history)

        # Define maximum lateral bound
        max_y_bound = 2.0  # Maximum lateral deviation in meters when steering is extreme
        y_bound = max_y_bound * avg_steering_angle  # Scale bound based on averaged steering

        # Calculate pixel-based bounds
        middle_x = self.view_width // 2
        pixel_y_bound = int((y_bound / max_y_bound) * (self.view_width // 2))

        # Define key points for the straight baseline curve
        bottom_y = self.view_height
        top_y = self.view_height // 2
        middle_y = (bottom_y + top_y) // 2

        # Adjust control points based on steering direction
        curve_width = 30  # Distance between left and right tube bounds
        bottom_width = 250

        # Define points for the left and right boundaries (start with straight lines)
        start_left = (middle_x - curve_width - bottom_width, bottom_y)
        start_right = (middle_x + curve_width + bottom_width, bottom_y)

        # For straight path, we will keep control points aligned to create straight lines
        control_left = (middle_x - curve_width, middle_y)
        control_right = (middle_x + curve_width, middle_y)

        # Adjust control points for steering to create a slight curve
        if avg_steering_angle < 0:  # Left turn
            control_left = (middle_x - curve_width + abs(pixel_y_bound), middle_y)
            control_right = (middle_x + curve_width + abs(pixel_y_bound), middle_y)
        elif avg_steering_angle > 0:  # Right turn
            control_left = (middle_x - curve_width - abs(pixel_y_bound), middle_y)
            control_right = (middle_x + curve_width - abs(pixel_y_bound), middle_y)

        end_left = (middle_x - curve_width + pixel_y_bound, top_y)
        end_right = (middle_x + curve_width + pixel_y_bound, top_y)

        # Generate points along the curves for the left and right boundaries
        self.left_motion_tube_points = self.generate_curve_points(start_left, control_left, end_left)
        self.right_motion_tube_points = self.generate_curve_points(start_right, control_right, end_right)
   
    # ===============================================================================================
    # Agent logic
    # ===============================================================================================

    def run_step(self):
        """Execute one step of navigation."""

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()

        # Check if the vehicle is affected by other vehicles. 
        # =====================================================================================================
        # Calculate the distance at which the vehicle should react        
        # max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed
        max_vehicle_distance = 30

        # Get RGB and depth image
        affected_by_vehicle = False

        try:
            # Extract the images from the dictionary
            self.RGB_image = self._image_dictionary["rgb_image"]
            self.depth_image = self._image_dictionary["bgr_depth"]
            self.depth_logarithmic = self._image_dictionary["log_depth"]

            if self.RGB_image is not None and self.depth_image is not None:
                # Get all the boundingboxes and distances of actors in the area
                self.spotted_actors = self._object_detector.extract_objects_from_surrounding(self.RGB_image, self.depth_image)
                # Check if vehicle is in the range of concern
                affected_by_vehicle, vehicle, distance = self._vehicle_obstacle_detected(self.spotted_actors, max_vehicle_distance)

            else:
                affected_by_vehicle = False
                print("No RGB_image or depth_image")
                
        except Exception as exc:
            raise exc
        

        # Initial throttle and brake
        throttle = 0.
        brake = 0.

        # Check if there is a vehicle
        if affected_by_vehicle:

            # Determine throttle and brake usign the RL agent
            # ====================================================================================================
            # Create observations
           
            vehicle_speed = self._vehicle.get_velocity().length()   # Get current vehicle speed to calculate radius in which to check for cars/lights
            target_speed = self._vehicle.get_speed_limit()          # Get maximum allowed speed (from traffic signs)


            unnorm_obs = np.ndarray([vehicle_speed, target_speed, distance], dtype=np.float32) # Create unnormalized observations
            
            throttle, brake = self.control_ACC_agent.predict(unnorm_obs)    # Get result from ACC controller

            # ====================================================================================================

        # The longitudinal PID is overwritten by the constant velocity but it is
        # still useful to apply it so that the vehicle isn't moving with static wheels
        control = self._local_planner.run_step()
        
        # Overwrite the throttle and brake based on what has happened
        control.throttle = throttle
        control.brake = brake
        
        return control

    def destroy_sensor(self):
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None

    def _vehicle_is_within_tubes(self, bbox):
        # We loop over all the motion tube point pairs and check wether a part of the vehicles boundingbox is inside
        x_min, y_min, x_max, y_max = map(int, bbox)
        is_within_bounds = False
        
        for left_point, right_point in zip(self.left_motion_tube_points, self.right_motion_tube_points):
            # Extract eh differnt x values and the y value (is the same)
            x_value_left, y_value = left_point
            x_value_right, _ = right_point
            
            if y_min < y_value < y_max:
                if x_min < x_value_left < x_max or x_min < x_value_right < x_max:
                    is_within_bounds = True
                    break

        return is_within_bounds


    def _vehicle_obstacle_detected(self, vehicle_list, max_vehicle_distance):
        """
        Check if there are vehicles to be concerned about
        """
        # Initialise variables
        smallest_distance = max_vehicle_distance
        is_vehicle_close = False
        closest_vehicle = None

        # Create motion tubes
        self._calculate_motion_tube()

        # If there are vehicles, check the closest vehicle in the motion tubes
        if len(vehicle_list) > 0:
            for label, distance, bbox in vehicle_list:
                if distance < smallest_distance and self._vehicle_is_within_tubes(bbox):
                    print(f"{label} at {distance} meter at position {bbox}")
                    is_vehicle_close = True
                    closest_vehicle = label
                    smallest_distance = distance

        return is_vehicle_close, closest_vehicle, smallest_distance


    def destroy(self):
        """Function to correctly destroy the agent"""
        self.rgb_camera.stop()
        self.rgb_camera.destroy()
        self.depth_camera.stop()
        self.depth_camera.destroy()
        self._vehicle.destroy()

    def respawn(self):
        """Respawn vehicle on different location"""
        location = random.choice(self._world.get_map().get_spawn_points())
        self._vehicle.set_transform(location)


    # ================================================================================================
    # Getters and Setters
    # ================================================================================================
    def get_RGB_image(self):
        return self.RGB_image

    def get_depth_image(self):
        return self.depth_logarithmic
    
    def get_spotted_actors(self):
        return self.spotted_actors

    def get_motion_tube_points(self):
        return self.left_motion_tube_points, self.right_motion_tube_points

   