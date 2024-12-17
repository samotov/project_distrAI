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
import numpy as np


from spawn import Spawn
from depth_conversion import to_bgra_array, depth_to_array
from models import ObjectLocalizationModel
from agents.basic_agent import BasicAgent

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
MAX_SIMULATION_LENGTH = 300
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
        self.reward = 0
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

    # Get the blueprints for the RGB camera based on a filter
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

    # Get the blueprints for the depth camera based on a filter
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

    # Change the synchronous mode of the world (True or Flase)
    def set_synchronous_mode(self, synchronous_mode):
        """
        Sets synchronous mode.
        """

        settings = self.world.get_settings()
        settings.synchronous_mode = synchronous_mode
        self.world.apply_settings(settings)

    # Setup the actor car vehicle that will be use in the environment
    def setup_car(self):
        """
        Spawns actor-vehicle to be controled.
        """

        car_bp = self.world.get_blueprint_library().filter('vehicle.*')[0]
        location = random.choice(self.world.get_map().get_spawn_points())
        self.car = self.world.try_spawn_actor(car_bp, location)
        while self.car == None:     # Try to spawn again if the spawn failed because of a collision
            print("Could not spawn car because of collision, trying again somehere else!")
            location = random.choice(self.world.get_map().get_spawn_points())
            self.car = self.world.try_spawn_actor(car_bp, location)
                
    # Setup the collision sensor
    def setup_collision_sensor(self):
        """
        Sets up collision sensor.
        """
        collision_sensor_bp = self.world.get_blueprint_library().find('sensor.other.collision')
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.collision_sensor.listen(lambda event: self._on_collision(event))

    # Setup the lane invasion sensor
    def setup_lane_invasion_sensor(self):  #####
        """
        Sets up lane invasion sensor.
        """
        lane_invasion_sensor_bp = self.world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.lane_invasion_sensor = self.world.spawn_actor(lane_invasion_sensor_bp, carla.Transform(), attach_to=self.car)
        weak_self = weakref.ref(self)
        self.lane_invasion_sensor.listen(lambda event: self._on_lane_invasion(event))

    # Function that will be called when a collision is detected
    def _on_collision(self, event):  #####
        """
        Collision event handler.
        """
        self.collision_hist.append(event)

    # Function that will be called when a lane invasion happens
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
            
    def render_bounding_boxes(self, display, results, depth_image):
        """
        Renders bounding boxes and their associated depth information on the display.
        """

        # Extract bounding box coordinates and labels
        bbox_array = results[0].boxes.xyxy.to('cpu').numpy()  # (N, 4) array of bounding box coordinates
        labels = results[0].boxes.cls.to('cpu').numpy()  # (N,) array of class labels

        # Extract the BGRA image and convert it to a grayscale depth map
        #bgra_image_array = np.array(depth_image.raw_data, dtype=np.uint8).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
        bgra_image_array = to_bgra_array(depth_image)
        grayscale_image = depth_to_array(bgra_image_array)

        # Prepare a font for drawing text
        font = pygame.font.Font(None, 16)

        # Initialize a list for output data
        data = []

        # Loop over bounding boxes and process each one
        for i, bbox in enumerate(bbox_array):
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = bbox.astype(int)
            label = int(labels[i])
            class_name = self.class_names[label] 
            
            x, y, depth, angle = self.get_depth_information(bbox)

            data.append({"label": label, "distance": depth})

            # Draw bounding box on the display
            pygame.draw.rect(display, (0, 255, 255), pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min), 2)
            
            # Render the text with a black frame (slightly offset for the shadow effect)
            shadow_surface = font.render(f"{class_name}, {x:.2f}m, {y:.2f}m, {depth:.2f}m, {angle:.2f}°", True, (0, 0, 0))
            display.blit(shadow_surface, (x_min - 1, y_min - 21))  # Slight offset for shadow
            display.blit(shadow_surface, (x_min + 1, y_min - 19))  # Slight offset for shadow
            display.blit(shadow_surface, (x_min, y_min - 20))      # Center for thicker effect

            # Display label and distance
            label_surface = font.render(f"{class_name}, {x:.2f}m, {y:.2f}m, {depth:.2f}m, {angle:.2f}°", True, (255, 255, 255))
            display.blit(label_surface, (x_min, y_min - 20))

        return data
    
    # Render the RL informatio in the data_dict to the top left of the screen
    def render_RL_information(self):
        # Prepare the data_dict dictionaty
        data_dict = dict()
        
        data_dict["speed"] = round(self.ego_speed, 2)
        data_dict["target speed"] = round(self.target_speed, 2)
        data_dict["distance to lead"] = round(self.distance_to_lead, 2)
        data_dict["reward"] = round(self.reward, 2)
        data_dict["actions"] = [round(self.throttle, 2), round(self.brake, 2)]
        
        # Initialize the font
        font = pygame.font.Font(None, 16)

        label_surface = font.render("RL data: ", True, TEXT_COLOR)
        self.display.blit(label_surface, (0, 10))

        # Draw each key and value pair on the top lef of the screen while incrementing the y position by 10
        position = 20
        
        for key in data_dict.keys():
            data_string = key + ": " + str(data_dict[key])
            label_surface = font.render(data_string, True, TEXT_COLOR)
            self.display.blit(label_surface, (0, position))
            position += 10
    # ------------------------------------ END: Display rendering methods-----------------------------------------------------------------

    # get depth information of a boundingbox in the image (x_top_view, y_top_view, depth_in_meters, angle)
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
        
        x_top_view = np.cos(angle * np.pi/180) * depth_in_meters
        y_top_view = np.sin(angle * np.pi/180) * depth_in_meters

        return [x_top_view, y_top_view, depth_in_meters, angle]

    # Update the RL input data based on the new environment
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
                
                _, y, depth, _ = self.get_depth_information(bbox)
                
                if depth < closest_vehicle_distance_in_range and abs(y) < Y_BOUND :  # Update, if this vehicle is closer then the previous one and is withing the y-bounds!
                    closest_vehicle_distance_in_range = depth # extract the depth to the lead vehicle so we can use it in the RL agent!
        
        # These 3 values are used for the step method!
        self.ego_speed = self.car.get_velocity().length() * 3.6
        self.target_speed = self.car.get_speed_limit()
        self.distance_to_lead = closest_vehicle_distance_in_range

    # Destroy the car and sensors in the environment
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

    # -------------------------------------------- START:Gym implementation ------------------------------------------------------------------
    
    # Implement a seed for the environment based on seed
    def seed(self, seed=None):
        """Set the random seed for reproducibility."""
        self.np_random, _ = gym.utils.seeding.np_random(seed)
    
    
    # Reset the environment
    def reset(self, **kwargs):
        
        """Reset the environment."""
        # Remove all the actors from the environment
        self.destroy_car_and_sensors()
        self.spawn.reomve_all_entities()

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
        self.destination = random.choice(self.spawn_points).location
        self.basic_agent.set_destination(self.destination)        

        # Reset the total timesteps
        self.total_timesteps = 0

        # Spawn new entities
        self.spawn.SpawnEntities(num_vehicles=20,num_walkers=30)
        
        # Return the initial observation
        return np.array([self.ego_speed, self.target_speed, self.distance_to_lead]), {}
    
    # Perform a step update in the environment based on an action
    def step(self, action):

        # Update the destination if it has been reached
        if self.basic_agent.done():
            self.basic_agent.set_destination(random.choice(self.spawn_points).location)
        
        # Apply the throttle and brake acoordint to action and other controls based on the basic agent for steer
        throttle, brake = np.clip(action, self.action_space.low, self.action_space.high)
        control = self.basic_agent.run_step()
        self.throttle = float(throttle)
        self.brake = float(brake)
        control.throttle = self.throttle
        control.brake = self.brake       
        self.car.apply_control(control)

        # We update the world simulation and capture an RGB and depth image
        self.world.tick()
        self.capture = True
        self.capture_depth = True
        self.total_timesteps += 1

        # Update YOLO detections and bounding boxes
        if self.image and self.depth_image:
            temp_image = np.array(self.image.raw_data).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
            self.results = self.model.forward(temp_image)

        # Update the RL data
        self.update_RL_input_data()

        # We render the environment
        self.render()

        self.reward = 1 - abs(self.ego_speed - self.target_speed)/self.target_speed
    

        truncated = False
        terminated = False
        if self.total_timesteps >= MAX_SIMULATION_LENGTH:
            terminated = True
        
        # Return observation, reward, terminated, truncated, and info
        return np.array([self.ego_speed, self.target_speed, self.distance_to_lead], dtype=np.float32), self.reward, terminated, truncated, {}
    
    # Render the environment
    def render(self):
        self.render_display()
        # self.render_bounding_boxes()
        self.render_RL_information()
        pygame.display.flip()
