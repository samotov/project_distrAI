import glob
import os
import sys
import psutil

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ========================================================================
# -- Environment Variables and CPU Thread Limitation ---------------------
# ========================================================================
# Limit the number of threads for CARLA
os.environ["OMP_NUM_THREADS"] = "6"

# Limit the number of threads for PyTorch (YOLO)
os.environ["MKL_NUM_THREADS"] = "3"
os.environ["OMP_NUM_THREADS"] = "3"
import torch
torch.set_num_threads(2)

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc
from depth_conversion import depth_to_logarithmic_grayscale, to_rgb_array, to_bgra_array, depth_to_array
from models import ObjectLocalizationModel

import argparse
import weakref
import random
import cv2
import time
import argparse
import textwrap

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_SPACE
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_TAB
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_p
    from pygame.locals import K_c
    from pygame.locals import K_l
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# Define symbolic parameters

VIEW_WIDTH = 1920 // 2
VIEW_HEIGHT = 1080 // 2
VIEW_FOV = 90
DISTANCE_NORMALIZER = 75
ENTITY_INPUT_AMOUNT = 3


max_distance = 1000

SENSOR_TICK = 0.05  # Sensor tick time in seconds
BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)
vehicle_bbox_record = True
pedestrian_bbox_record = False
count = 0

rgb_info = np.zeros((540, 960, 3), dtype="i")
seg_info = np.zeros((540, 960, 3), dtype="i")

# CPU affinity function
def set_cpu_affinity(pid, cores):
    """
    Sets CPU affinity for a given process.
    :param pid: Process ID
    :param cores: List of CPU cores to bind to
    """
    try:
        process = psutil.Process(pid)
        process.cpu_affinity(cores)
        print(f"Process {pid} pinned to CPUs: {cores}")
    except Exception as e:
        print(f"Error setting CPU affinity for process {pid}: {e}")


# ==============================================================================
# -- BasicSynchronousClient ----------------------------------------------------
# ==============================================================================

class BasicSynchronousClient(object):
    """
    Basic implementation of a synchronous client.
    """

    def __init__(self):

        self.client = None
        self.world = None
        self.camera = None

        self.depth_camera = None
        self.collision_sensor = None  # add collision sensor
        self.lane_invasion_sensor = None  ####
        self.car = None

        self.display = None
        self.image = None
        self.depth_image = None

        # Capture
        self.capture = True
        self.capture_depth = True
        
        # Record
        self.record = True
        self.rgb_record = False
        self.depth_record = False

        self.model = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')
        
        self.collision_hist = []
        self.lane_invasion_hist = []


        self.loop_state = False
        self.class_names = ["Car", "Motorcycle", "truck","Pedestrian","Bus","Stop sign","Green","Orange","Red", "Not important"]

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
        self.car = self.world.spawn_actor(car_bp, location)

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

    def control(self, car):
        """
        Applies control to main car based on pygame pressed keys.
        Will return True If ESCAPE is hit, otherwise False to end main loop.
        """
        keys = pygame.key.get_pressed()

        if keys[K_ESCAPE]:
            return True

        control = car.get_control()
        control.throttle = 0
        if keys[K_w]:
            control.throttle = 1
            control.reverse = False
        elif keys[K_s]:
            control.throttle = 1
            control.reverse = True
        if keys[K_a]:
            control.steer = max(-1., min(control.steer - 0.05, 0))
        elif keys[K_d]:
            control.steer = min(1., max(control.steer + 0.05, 0))
        else:
            control.steer = 0
        if keys[K_p]:
            car.set_autopilot(True)
        if keys[K_l]:
            self.loop_state = True
        if keys[K_l] and (pygame.key.get_mods() & pygame.KMOD_SHIFT):
            self.loop_state = False
        control.hand_brake = keys[K_SPACE]

        car.apply_control(control)
        return False

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
            # cv2.imwrite('custom_data/image' + str(self.image_count) + '.png', i3)
            # print("RGB(custom)Image")


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
    

    def render(self, display):
        """
        Transforms image from camera sensor and blits it to main pygame display.
        """

        if self.image is not None:
            array = np.frombuffer(self.image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (self.image.height, self.image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            display.blit(surface, (0, 0))
            
    def render_bounding_boxes(self, display, results, depth_image):
        """
        Renders bounding boxes and their associated depth information on the display.
        """
        import numpy as np

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


    def retrieve_RL_input_data(self):
        bbox_array = self.results[0].boxes.xyxy.to('cpu').numpy()  # (N, 4) array of bounding box coordinates
        labels = self.results[0].boxes.cls.to('cpu').numpy()
        # ["Car", "Motorcycle", "truck","Pedestrian","Bus","Stop sign","Green","Orange","Red", "Not important"]

        

        # Caculate stop flag and distance
        data_dict = dict()
        
        stop_amounts = 0
        green_amounts = 0
        minimal_distance = DISTANCE_NORMALIZER
        entity_xy_array = list()
        for i, bbox in enumerate(bbox_array):
            x, y, depth, _ = self.get_depth_information(bbox)
            label = labels[i]

            if label in [0, 1, 2, 3, 4]:
                entity_xy_array.append([x, y, depth])
            elif label in [5, 7, 8]:
                stop_amounts += 1
                minimal_distance = min(minimal_distance, x)
            elif label == 6:
                green_amounts += 1        

        if stop_amounts > green_amounts:
            stopping_distance_norm = minimal_distance/DISTANCE_NORMALIZER
        else:
            stopping_distance_norm = 1
        
        if len(entity_xy_array) != 0:
            entity_xy_array.sort(key=lambda x:x[2])

            entities = list()
            for i in range(ENTITY_INPUT_AMOUNT):
                if i < len(entity_xy_array):
                    entities.append([entity_xy_array[i][0]/DISTANCE_NORMALIZER, entity_xy_array[i][1]/DISTANCE_NORMALIZER])
                else:
                    entities.append([1, 0])
        else:
            entities = [[1, 0], [1, 0], [1, 0]]
                
        
        

        # Get the steering angle
        steer_angle_FL = self.car.get_wheel_steer_angle(carla.VehicleWheelLocation(0))
        steer_angle_FR = self.car.get_wheel_steer_angle(carla.VehicleWheelLocation(1))

        data_dict["Velocity"] = self.car.get_velocity().length()
        data_dict["Speed limit"] = self.car.get_speed_limit()
        data_dict["Steer angle"] = (steer_angle_FL + steer_angle_FR)/2      # Use the average angle of the front steering wheels
        data_dict["Stopping distance normalized"] = stopping_distance_norm
        data_dict["Closest entities"] = entities

        return data_dict



    def print_data(self, data_dict, color):
        font = pygame.font.Font(None, 16)

        label_surface = font.render("RL data: ", True, color)
        self.display.blit(label_surface, (0, 10))

        position = 20
        for key in data_dict.keys():
            if key == "Closest entities":
                data_string = key + ": " + str([[round(elem, 2) for elem in entity] for entity in data_dict[key]])
            else:
                data_string = key + ": " + str(data_dict[key])
            
            label_surface = font.render(data_string, True, color)
            self.display.blit(label_surface, (0, position))
            position += 10


    def game_loop(self, args):
        """
        Main program loop.
        """
        try:
            pygame.init()

            self.client = carla.Client('127.0.0.1', 2000)
            self.client.set_timeout(2.0)
            self.world = self.client.get_world()

            self.setup_car()
            self.setup_camera()
            self.setup_depth()

            self.display = pygame.display.set_mode((VIEW_WIDTH, VIEW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame_clock = pygame.time.Clock()

            self.set_synchronous_mode(True)


            while True:
                self.world.tick()
                self.capture = True
                self.capture_depth = True

                pygame_clock.tick_busy_loop(60)

                # Render RGB image
                self.render(self.display)

                # Update YOLO detections and bounding boxes
                if self.image and self.depth_image:
                    temp_image = np.array(self.image.raw_data).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
                    self.results = self.model.forward(temp_image)
                    # self.render_bounding_boxes(self.display, self.results, self.depth_image)
                    RL_data_dict = self.retrieve_RL_input_data()
                    self.print_data(RL_data_dict, (255, 0, 0))

                pygame.display.flip()

                # Handle user inputs
                pygame.event.pump()

                if self.control(self.car):
                    return
                
                

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.depth_camera.destroy()
            self.car.destroy()
            pygame.quit()



# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================

def main():
    """
    Initializes the client-side bounding box demo.
    """
    
     # Print the number of available CPU cores
    cpu_count = os.cpu_count()
    print(f"Number of available CPU cores: {cpu_count}")
    
     # Step 1: Set CPU affinity for the CARLA process
    carla_cores = [0, 1, 2, 3]  # Assign CARLA to specific CPU cores
    current_pid = os.getpid()  # Get current process PID
    set_cpu_affinity(current_pid, carla_cores)

    # Step 3: Pin YOLO processing to separate CPU cores (optional, if GPU is handling inference)
    yolo_cores = [4, 5, 6, 7,8]  # Optionally assign YOLO logic to specific CPU cores for preprocessing
    set_cpu_affinity(current_pid, yolo_cores)
    
    
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=100,
        type=int,
        help='set Capture Cycle settings, Recommend : above 100')

    args = argparser.parse_args()

    print(__doc__)

    try:
        client = BasicSynchronousClient()
        client.game_loop(args)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
