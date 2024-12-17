import gymnasium as gym
import os
import carla
import weakref
import random
import cv2
import pygame
import torch
import sys
import glob
import math
import numpy as np


from spawn import Spawn
from depth_conversion import to_bgra_array, depth_to_array
from models import ObjectLocalizationModel
from agents.basic_agent import BasicAgent
from collections import deque

# System stuff
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# Threads number limitation
os.environ["OMP_NUM_THREADS"] = "6"     # carla
os.environ["MKL_NUM_THREADS"] = "3"     # pytorch (YOLO)
os.environ["OMP_NUM_THREADS"] = "3"
torch.set_num_threads(2)


# Define symbolic parameters
VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90
Y_BOUND = 1 # 1 * 2 = 2: width in which we look for the lead vehicle 
MAX_SIMULATION_LENGTH = 1000
max_distance = 1000
SENSOR_TICK = 0.03  # Sensor tick time in seconds
BB_COLOR = (248, 64, 24)
TEXT_COLOR = (255, 0, 0)



class CarlaEnv(gym.Env):

    """
    Custom Gym environment for CARLA simulation.
    """

    def __init__(self):
        super(CarlaEnv, self).__init__()
        self.np_random = None  # For reproducibility
        
        # Action space: throttle and brake only (continuous between 0.0 and 1.0)
        self.action_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1.0, 1.0]))

        # Observation space: ego_speed, target_speed, distance_to_lead (continuous values)
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Minimum values
            high=np.array([50.0, 50.0, 100.0], dtype=np.float32),  # Maximum values (adjust as needed)
            )
                
        # Environment simulition intialization
        self.client = carla.Client('127.0.0.1', 2000)
        self.client.set_timeout(2.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()
        self.spawn = Spawn()
        self.set_synchronous_mode(False)
        self.spawn_points = self.map.get_spawn_points()

        # RL network data (intialization)
        self.ego_speed = 0
        self.target_speed = 0
        self.distance_to_lead =  100
        self.target_speed_based_on_distance = 0
        self.throttle = 0
        self.brake = 0

        # Reward function parameters
        self.safe_distance = 5 # minimal safe following distance at target speed
        self.target_speed_range = 2.78 # 10 km/h range around the target speed 

        # Initialize the object detection model
        self.model = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')

        self.camera = None
        self.depth_camera = None
        self.collision_sensor = None 
        self.lane_invasion_sensor = None
        self.car = None

        # Initialize pygame and the display
        pygame.init()
        self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.image = None
        self.depth_image = None
        

        # Capture
        self.capture = True
        self.capture_depth = True
        
        # Record
        self.record = True
        self.rgb_record = False
        self.depth_record = False

        
        self.collision_hist = []
        self.lane_invasion_hist = []
        
       
        self.loop_state = False
        self.class_names = ["Car", "Motorcycle", "truck","Pedestrian","Bus","Stop sign","Green","Orange","Red", "Not important"]
        
        # Initialize a deque to store the last 10 steering angles
        self.steering_angle_history = deque(maxlen=10)
        self.timestamp_history = deque(maxlen=10)
        
        # Initial top bound positions for the perspective effect
        self.top_left_offset = -50
        self.top_right_offset = 50
        
        self.left_motion_tube_points = []
        self.right_motion_tube_points = []
        
        self.lead_car = None
        

    def camera_blueprint(self, filter):
        """
        Returns camera blueprint.
        """
        camera_bp = self.world.get_blueprint_library().find(filter)
        camera_bp.set_attribute('image_size_x', str(VIEW_WIDTH))
        camera_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))
        camera_bp.set_attribute('fov', str(VIEW_FOV))
        camera_bp.set_attribute('sensor_tick', str(SENSOR_TICK))  # Time interval for sensor updates
        return camera_bp


    def depth_blueprint(self, filter):

        """
        Returns depth blueprint with specified symbolic attributes.
        """
        depth_bp = self.world.get_blueprint_library().find(filter)

        # Set depth sensor-specific attributes using symbolic parameters
        depth_bp.set_attribute('image_size_x', str(VIEW_WIDTH))  # Width of the output depth image
        depth_bp.set_attribute('image_size_y', str(VIEW_HEIGHT))  # Height of the output depth image
        depth_bp.set_attribute('fov', str(VIEW_FOV))  # Field of view of the depth sensor
        depth_bp.set_attribute('sensor_tick', str(SENSOR_TICK))  # Time interval for sensor updates


        return depth_bp

    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car == None:
            print("Could not spawn car because of collision, trying again somehere else!")
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.try_spawn_actor(car_bp, location)
        
        # Spawn lead vehicle
        lead_car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        forward_dir = self.car.get_transform().get_forward_vector() * 10
        location.location += carla.Location(forward_dir.x, forward_dir.y, forward_dir.z)
        self.lead_car = self.world.try_spawn_actor(lead_car_bp, location)

    def setup_collision_sensor(self):  #####
        """
        Sets up collision sensor.
        """
        collision_sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    def setup_lane_invasion_sensor(self):  #####
        """
        Sets up lane invasion sensor.
        """
        lane_invasion_sensor_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_sensor_bp, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    def _on_collision(self, event):  #####
        """
        Collision event handler.
        """
        self.collision_hist.append(event)

    def _on_lane_invasion(self, event):  #####
        """
        Lane invasion event handler.
        """
        self.lane_invasion_hist.append(event)

    # ------------------------------------ START: Setup depth camera methods-------------------------------------------------------------
    def setup_depth(self):
        # Depth camera
        depth_transform = carla.Transform(carla.Location(x=1.6, z=1), carla.Rotation(pitch=0))
        self.depth_camera = self.world.spawn_actor(self.depth_blueprint('sensor.camera.depth'), depth_transform,
                                                   attach_to=self.car)
        weak_self = weakref.ref(self)
        self.depth_camera.listen(lambda depth_image: weak_self().set_depth(weak_self, depth_image))

        # Calibration
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))

        self.depth_camera.calibration = calibration

    

    @staticmethod
    def set_depth(weak_self, depth_img): # takes a bbox list (returned from the neural network) as an input to calculate the nearest/mean pixel within each bbox
        """
        Extracts labels and their corresponding distances.
        """
        data = []
        self = weak_self()
        if self.capture_depth:
            self.depth_image = depth_img
            self.capture_depth = False

        if self.depth_record:
            i = np.array(depth_img.raw_data, dtype=np.uint8)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))


            bgra_image_array = i2[:, :, :3] # BGRA image
            grayscale_image = depth_to_array(bgra_image_array) # logarithmic depth image (we use this image to find the darkest/mean spot within each bbox)

            bbox_list = self.results[0].boxes.xyxy.to('cpu').tolist()

            for bbox in bbox_list: # go over every bouningbox in the list
                
                # Convert bbox coordinates to integers
                x_min, y_min, x_max, y_max = map(int, bbox)
                
                # Find closest depth value in bounding box
                closest_depth = float('inf')
                closest_coord = None

                for y in range(y_min -1 , y_max - 1):
                        for x in range(x_min - 1, x_max - 1):
                            depth_value = grayscale_image[y, x, 0]
                            if depth_value < closest_depth:
                                closest_depth = depth_value
                                closest_coord = (x, y)
                
                if closest_coord:   
                    B, G, R = bgra_image_array[closest_coord[1], closest_coord[0]] # Assuming bgra_image_array is in BGRA format
                    normalized = (int(R) + int(G) * 256 + int(B) * 256 * 256) / (256 * 256 * 256 - 1)
                    in_meters = max_distance * normalized
    
    # ------------------------------------ END: Setup depth camera methods-------------------------------------------------------------

    # ------------------------------------ START: Setup RGB camera methods-------------------------------------------------------------
    def setup_camera(self):
        """
        Spawns actor-camera to be used to render view.
        Sets calibration for client-side boxes rendering.
        """

        # RGB camera
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1), carla.Rotation(pitch=0))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform,
                                             attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        # Calibration
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration


    @staticmethod
    def set_image(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """
        self = weak_self()
        if self.capture:
            self.image = img
            self.capture = False

        if self.rgb_record:
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]

            image_bgr = cv2.cvtColor(i3*255, cv2.COLOR_RGB2BGR)
            temp_image_path = "temp_image.png"
            cv2.imwrite(temp_image_path, image_bgr)
            self.results = self.testNetwork(i3)

    # ------------------------------------ END: Setup RGB camera methods-----------------------------------------------------------------

    # ------------------------------------ START: display rendering methods -------------------------------------------------------------
    def render_display(self):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))

    def render_bounding_boxes(self):
        """
        Renders bounding boxes and their associated depth information on the display.
        """

        # Extract bounding box coordinates and labels
        bbox_array = self.results[0].boxes.xyxy.to('cpu').numpy()  # (N, 4) array of bounding box coordinates
        labels = self.results[0].boxes.cls.to('cpu').numpy()  # (N,) array of class labels

        # Prepare a font for drawing text
        font = pygame.font.Font(None, 16)

        # Loop over bounding boxes and process each one
        for i, bbox in enumerate(bbox_array):
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = bbox.astype(int)
            label = int(labels[i])
            class_name = self.class_names[label] 
            
            x, y, depth, angle = self.get_depth_information(bbox)

            if self.bbox_is_within_tubes(bbox):
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            # Draw bounding box on the display
            pygame.draw.rect(self.display, color, pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min), 2)
            
            # Render the text with a black frame (slightly offset for the shadow effect)
            shadow_surface = font.render(f"{class_name}, {x:.2f}m, {y:.2f}m, {depth:.2f}m, {angle:.2f}°", True, (0, 0, 0))
            self.display.blit(shadow_surface, (x_min - 1, y_min - 21))  # Slight offset for shadow
            self.display.blit(shadow_surface, (x_min + 1, y_min - 19))  # Slight offset for shadow
            self.display.blit(shadow_surface, (x_min, y_min - 20))      # Center for thicker effect

            # Display label and distance
            label_surface = font.render(f"{class_name}, {x:.2f}m, {y:.2f}m, {depth:.2f}m, {angle:.2f}°", True, (255, 255, 255))
            self.display.blit(label_surface, (x_min, y_min - 20))
    
    def render_RL_information(self):
        data_dict = dict()
        
        data_dict["speed"] = round(self.ego_speed, 2)
        data_dict["target speed"] = round(self.target_speed, 2)
        data_dict["distance to lead"] = round(self.distance_to_lead, 2)
        data_dict["reward"] = round(self.reward, 2)
        data_dict["actions"] = [round(self.throttle, 2), round(self.brake, 2)]
        data_dict["breaking distance speed"] = round(self.target_speed_based_on_distance, 2)


        font = pygame.font.Font(None, 16)

        label_surface = font.render("RL data: ", True, TEXT_COLOR)
        self.display.blit(label_surface, (0, 10))

        position = 20
        
        for key in data_dict.keys():
            data_string = key + ": " + str(data_dict[key])
            label_surface = font.render(data_string, True, TEXT_COLOR)
            self.display.blit(label_surface, (0, position))
            position += 10
            
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
                    
            
    def draw_motion_tube(self):
        # Get the current steering angle
        current_steering_angle = self.car.get_control().steer  # Value between -1.0 (left) and 1.0 (right)

        # Add the current steering angle to the history
        self.steering_angle_history.append(current_steering_angle)

        # Compute the average steering angle over the last 10 frames
        if len(self.steering_angle_history) > 20:
            self.steering_angle_history.pop(0)
        avg_steering_angle = sum(self.steering_angle_history) / len(self.steering_angle_history)

        # Define maximum lateral bound
        max_y_bound = 2.0  # Maximum lateral deviation in meters when steering is extreme
        y_bound = max_y_bound * avg_steering_angle  # Scale bound based on averaged steering

        # Calculate pixel-based bounds
        middle_x = VIEW_WIDTH // 2
        pixel_y_bound = int((y_bound / max_y_bound) * (VIEW_WIDTH // 2))

        # Define key points for the straight baseline curve
        bottom_y = VIEW_HEIGHT
        top_y = VIEW_HEIGHT // 2
        middle_y = (bottom_y + top_y) // 2

        # Adjust control points based on steering direction
        curve_width = 35  # Distance between left and right tube bounds
        bottom_width = 300

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

        # Draw the motion tube as curves
        line_color = (0, 0, 255)  # Blue color for the bounds
        for i in range(len(self.left_motion_tube_points) - 1):
            pygame.draw.line(self.display, line_color, self.left_motion_tube_points[i], self.left_motion_tube_points[i + 1], 2)
            pygame.draw.line(self.display, line_color, self.right_motion_tube_points[i], self.right_motion_tube_points[i + 1], 2)

    # ------------------------------------ END: Display rendering methods-----------------------------------------------------------------

    def get_depth_information(self, bbox):
        x_min, y_min, x_max, y_max = bbox.astype(int)

        # Use the median depth in from the boundingbox as the depth
        bgra_image_array = to_bgra_array(self.depth_image)
        grayscale_image = depth_to_array(bgra_image_array)
        cropped_depth = grayscale_image[max(0, y_min):min(VIEW_HEIGHT, y_max), max(0, x_min):min(VIEW_WIDTH, x_max)]

        if cropped_depth.size > 0:
            median_depth = np.median(cropped_depth)

            # Identify the coordinate closest to the median depth value
            abs_diff = np.abs(cropped_depth - median_depth)
            median_coord = np.unravel_index(abs_diff.argmin(), cropped_depth.shape)

            # Compute distance in meters
            bgra_image_array = to_bgra_array(self.depth_image)
            B, G, R = bgra_image_array[
                median_coord[0] + max(0, y_min), median_coord[1] + max(0, x_min), :3
            ]
            normalized = (int(R) + int(G) * 256 + int(B) * 256 * 256) / (256 **3 - 1)

            depth_in_meters = max_distance * normalized
        else:
            depth_in_meters = float('nan')
        
        # Calculate the 2D angle to an object
        middle_bbx = (x_max + x_min)/2

        w, _ = pygame.display.get_surface().get_size()

        x_distance = middle_bbx - w/2
        x_factor = x_distance / (w/2)
        FOV_constant = VIEW_FOV/2

        angle =  x_factor * FOV_constant
        
        
        # Position of the median point
        x_top_view = np.cos(angle * np.pi/180) * depth_in_meters
        y_top_view = np.sin(angle * np.pi/180) * depth_in_meters

        return [x_top_view, y_top_view, depth_in_meters, angle]

    def bbox_is_within_tubes(self, bbox):
        # We loop over all the motion tube point pairs and check wether a part of the vehicles boundingbox is inside
        x_min, y_min, x_max, y_max = bbox.astype(int)
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
    
    def update_RL_input_data(self):
        
        bbox_array = self.results[0].boxes.xyxy.to('cpu').numpy()  # (N, 4) array of bounding box coordinates
        labels = self.results[0].boxes.cls.to('cpu').numpy()
        # ["Car", "Motorcycle", "truck","Pedestrian","Bus","Stop sign","Green","Orange","Red", "Not important"]
        # Find the closest vehicle, with a max value of 100
        closest_vehicle_distance_in_range = 100
        
        for i, bbox in enumerate(bbox_array):
            
            label = labels[i]
            
            # for now, we only look at vehicles ...
            if label in [0, 1, 2, 4]:
                 
                _, _, depth, _ = self.get_depth_information(bbox)

                if self.bbox_is_within_tubes(bbox) and depth < closest_vehicle_distance_in_range:
                    closest_vehicle_distance_in_range = depth  # Update the closest vehicle distance if it's within bounds
            
        # These 3 values are used for the step method!
        self.ego_speed = self.car.get_velocity().length() * 3.6
        self.target_speed = self.car.get_speed_limit()
        self.distance_to_lead = closest_vehicle_distance_in_range


    def destroy_car_and_sensors(self):
        # Destroy camera
        if self.camera != None:
            self.camera.destroy()
        
        # Destroy depth camera
        if self.depth_camera != None:
            self.depth_camera.destroy()
        
        # Destroy collision sensor
        if self.collision_sensor != None:
            self.collision_sensor.destroy()
        
        # Destroy lane invasion sensor
        if self.lane_invasion_sensor != None:
            self.lane_invasion_sensor.destroy()

        # Destroy car
        if self.car != None:
            self.car.destroy()   

        if self.lead_car != None:
            self.lead_car.destroy()    
            

    # -------------------------------------------- START:Gym implementation ------------------------------------------------------------------
    
    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)
    
    
    def reset(self, **kwargs):
        
        """Reset the environment."""
        # Remove all the actors from the environment
        self.destroy_car_and_sensors()
        self.spawn.remove_all_entities()

        # Update the seed
        seed = kwargs.get("seed", None)
        if seed is not None:
            self.seed(seed)
        # Setup a new car and cameras
        self.setup_car()
        self.setup_camera()
        self.setup_depth()

        # Initialize the basic agent that will take of the route planning and the destination
        self.basic_agent = BasicAgent(self.car,  map_inst=self.map)
        self.lead_basic_agent = BasicAgent(self.lead_car, map_inst=self.map)
        self.destination = random.choice(self.spawn_points).location
        self.basic_agent.set_destination(self.destination)        

        # Reset the total timesteps
        self.total_timesteps = 0

        # Spawn new entities
        self.spawn.SpawnEntities(num_vehicles=20,num_walkers=30)
        
        # Return the initial observation
        return np.array([self.ego_speed, self.target_speed, self.distance_to_lead]), {}
    
    
    def step(self, action):
        """
        Execute a single step in the environment based on the given action.

        Principles:
        - Avoid directly controlling `steer`, `throttle`, and `brake` in the environment.
        - Instead, use the action as the agent's command.

        Parameters:
        - action: The action provided by the agent.

        Returns:
        - obs: The current observation.
        - reward: The calculated reward for the current step.
        - terminated: Whether the episode has terminated.
        - truncated: Whether the episode has been truncated (e.g., due to reaching a time limit).
        - info: Additional diagnostic information.
        """

        # Clip the action to the defined action space range
        self.throttle, self.brake = np.clip(action, self.action_space.low, self.action_space.high)
        
        
        reward, terminated, truncated = self.calculate_reward()
        
        # --------------------------- Transform to calculate the distance between the ego and the lead ------------------------ #
        location_lead = self.lead_car.get_transform().location
        location_agent = self.car.get_transform().location
    
        distance = location_agent.distance(location_lead)
        lead_control = self.lead_basic_agent.run_step()
        if (distance > 200):
            self.lead_basic_agent.add_emergency_stop(lead_control)
        
        self.lead_car.apply_control(lead_control)

    # Calculate the Euclidean distance
        # Generate control commands based on the agent's action
        control = self.basic_agent.run_step()
        control.throttle = float(self.throttle)
        control.brake = float(self.brake)
        # If steering needs to be learned, include it in the action space and set `control.steer` here.

        # Apply the control commands to the vehicle
        self.car.apply_control(control)
        self.world.tick()
        self.capture = True
        self.capture_depth = True
        self.total_timesteps += 1

        # Process images and perform object detection
        if self.image and self.depth_image:
            temp_image = np.array(self.image.raw_data).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
            self.results = self.model.forward(temp_image)

        # Update state variables (ego_speed, target_speed, distance_to_lead)
        self.update_RL_input_data()

        # Render the scene (if necessary)
        self.render()
        
        reward, terminated, truncated = self.calculate_reward()

        # Combine observations into `obs`
        obs = np.array([self.ego_speed, self.target_speed, self.distance_to_lead], dtype=np.float32)
        info = {}

        # Return using the Gymnasium-style API: obs, reward, terminated, truncated, info
        return obs, reward, terminated, truncated, info
    

    def calculate_reward(self):
        """
        Calculate the reward for the current step based on:
        - Maintaining the target speed.
        - Maintaining a safe distance to the lead vehicle.
        - Avoiding collisions and extreme actions.
        """

        terminated = False
        truncated = False
        
        # -------------------- 1. Basic Reward: Maintain Target Speed --------------------
        
        reward = max(0, 1 - 0.5*(abs(self.ego_speed - self.target_speed) / (27.77 - self.target_speed)))

        # -------------------- 2. Collision Penalty --------------------
        if self.distance_to_lead < 1.0:  # Collision detected
            print("Collision detected!")
            reward = -1
            terminated = True
        # -------------------- 3. Distance-Based Reward --------------------
              
        optimal_distance = 10.0  # Optimal distance to maintain in meters
        safe_margin = 5.0        # Allowable margin around optimal distance

        # Reward for maintaining an optimal safe distance
        if abs(self.distance_to_lead - optimal_distance) <= safe_margin:
            reward += 0.5  # Reward for staying in the optimal range
        elif self.distance_to_lead < optimal_distance:  
            # Penalty for being too close to the lead vehicle
            reward -= 0.3 * (optimal_distance - self.distance_to_lead)
        elif self.distance_to_lead > optimal_distance + safe_margin:
            # Small penalty for being too far from the lead vehicle
            reward -= 0.1 * (self.distance_to_lead - (optimal_distance + safe_margin))


        # -------------------- 4. Safe Distance Bonus --------------------
        
        if self.distance_to_lead > 20.0:
            reward += 0.1
            
        # -------------------- 5. Intelligent Deceleration Reward --------------------
        
        safe_distance = 10.0
        deceleration_reward = 0.2
        deceleration_penalty = -0.5
        if self.distance_to_lead < safe_distance:
            # Check expected deceleration behavior
            expected_behavior = (self.throttle < 0.3 or self.brake > 0.5)
            if expected_behavior:
                reward += deceleration_reward
            else:
                reward += deceleration_penalty

        # -------------------- 6. Extreme Action Penalties --------------------
        
        extreme_action_penalty = -0.1
        if self.throttle > 0.9:
            reward += extreme_action_penalty
        if self.brake > 0.9:
            reward += extreme_action_penalty

        # -------------------- Termination Conditions --------------------
        
        if self.total_timesteps >= MAX_SIMULATION_LENGTH:
            terminated = True
        
        self.reward = reward
        # Return using the Gymnasium-style API: obs, reward, terminated, truncated, info
        return reward, terminated, truncated

            
    def render(self):
        self.render_display()
        self.draw_motion_tube()
        # self.render_bounding_boxes()
        self.render_RL_information()
        pygame.display.flip()
