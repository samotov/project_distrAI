
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
from depth_conversion import depth_to_logarithmic_grayscale, to_rgb_array
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

# MIN_RANGE = 0.5             # Minimum range for depth sensor in meters
# MAX_RANGE = 50.0            # Maximum range for depth sensor in meters
SENSOR_TICK = 0.1  # Sensor tick time in seconds
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

        self.loop_state = False

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

    def setup_depth(self):
        # Depth camera
        depth_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
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
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
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
        self = weak_self()
        if self.capture_depth:
            self.depth_image = depth_img
            self.capture_depth = False

        if self.depth_record:
            i = np.array(depth_img.raw_data, dtype=np.uint8)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))


            bgra_image_array = i2[:, :, :3] # BGRA image
            grayscale_image = depth_to_logarithmic_grayscale(bgra_image_array) # logarithmic depth image (we use this image to find the darkest/mean spot within each bbox)

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
                    normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                    in_meters = 1000 * normalized
                    print(f"Closest object at ({closest_coord}): Depth {in_meters}")
        
    def testNetwork(self, image_path): # test the Yolo network on image classification
        
        yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')
        results = yolomodel.forward(image_path)
        
        return results

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
        bbox_list = results[0].boxes.xyxy.to('cpu').tolist()
        labels = results[0].boxes.cls.to('cpu').tolist()

        for i, bbox in enumerate(bbox_list):
            # Bounding box coordinates
            x_min, y_min, x_max, y_max = map(int, bbox)
            label = f"Class: {int(labels[i])}"

            # Find the closest depth within the bounding box
            closest_depth = float('inf')
            closest_coord = None

            bgra_image_array = np.array(depth_image.raw_data, dtype=np.uint8).reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))[:, :, :3]
            grayscale_image = depth_to_logarithmic_grayscale(bgra_image_array)

            for y in range(max(0, y_min), min(VIEW_HEIGHT, y_max)):
                for x in range(max(0, x_min), min(VIEW_WIDTH, x_max)):
                    depth_value = grayscale_image[y, x, 0]
                    if depth_value < closest_depth:
                        closest_depth = depth_value
                        closest_coord = (x, y)

            # Compute distance in meters
            if closest_coord:
                B, G, R = bgra_image_array[closest_coord[1], closest_coord[0], :3]
                normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
                in_meters = 1000 * normalized
            else:
                in_meters = float('nan')

            # Draw bounding box on the display
            pygame.draw.rect(display, (0, 255, 0), pygame.Rect(x_min, y_min, x_max - x_min, y_max - y_min), 2)

            # Display label and distance
            font = pygame.font.Font(None, 24)
            label_surface = font.render(f"{label}, Depth: {in_meters:.2f}m", True, (255, 255, 255))
            display.blit(label_surface, (x_min, y_min - 20))

            

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
                    self.results = self.testNetwork(temp_image)
                    self.render_bounding_boxes(self.display, self.results, self.depth_image)

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
    
     # Set CPU affinity for the CARLA process
    carla_cores = [0, 1, 3, 4]  # Assign CARLA to cores 0 and 1
    current_pid = os.getpid()
    set_cpu_affinity(current_pid, carla_cores)

    # Import torch AFTER setting affinity for PyTorch thread control
    os.environ["OMP_NUM_THREADS"] = "3"
    os.environ["MKL_NUM_THREADS"] = "3"
    import torch
    torch.set_num_threads(2)

    # Pin YOLO (main Python process) to different cores
    yolo_cores = [5, 6, 7, 8]  # Assign YOLO to cores 2 and 3
    set_cpu_affinity(current_pid, yolo_cores)

    # The rest of your CARLA and YOLO implementation...
    print("CARLA and YOLO CPU affinity successfully set up.")
    
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
