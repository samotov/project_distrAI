#!/usr/bin/env python

"""
An example of client-side bounding boxes with basic car controls.

Controls:
Welcome to CARLA for Getting Bounding Box Data.
Use WASD keys for control.
    W            : throttle
    S            : brake
    AD           : steer
    Space        : hand-brake
    P            : autopilot mode
    C            : Capture Data
    l            : Loop Capture Start
    L            : Loop Capture End

    ESC          : quit
"""

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================

import carla

from carla import ColorConverter as cc
from depth_conversion import depth_to_logarithmic_grayscale, to_rgb_array
import test_object_localization_model

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

# LiDAR-specific parameters
MAX_LIDAR_RANGE = 100.0  # Maximum detection range for LiDAR in meters
ROTATION_FREQUENCY = 10.0  # Rotations per second for LiDAR
NUM_CHANNELS = 32  # Number of channels for LiDAR
POINTS_PER_SECOND = 56000  # Points per second for LiDAR

BB_COLOR = (248, 64, 24)
WBB_COLOR = (0, 0, 255)
vehicle_bbox_record = True
pedestrian_bbox_record = False
count = 0

rgb_info = np.zeros((540, 960, 3), dtype="i")
seg_info = np.zeros((540, 960, 3), dtype="i")


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
        self.camera_segmentation = None
        self.depth_camera = None
        self.car = None

        self.display = None
        self.image = None
        self.depth_image = None

        # Capture
        self.capture = True
        self.capture_segmentation = True
        self.capture_depth = True

        # Record
        self.record = True
        self.rgb_record = False
        self.depth_record = False

        self.screen_capture = 0
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

    def lidar_blueprint(self, filter):
        """
        Returns LiDAR blueprint with specified symbolic attributes.
        """
        lidar_bp = self.world.get_blueprint_library().find(filter)

        # Set LiDAR-specific attributes using symbolic parameters
        lidar_bp.set_attribute('range', str(MAX_LIDAR_RANGE))  # Max detection range in meters
        lidar_bp.set_attribute('rotation_frequency', str(ROTATION_FREQUENCY))  # Rotations per second
        lidar_bp.set_attribute('channels', str(NUM_CHANNELS))  # Number of channels (height of LiDAR scan)
        lidar_bp.set_attribute('points_per_second', str(POINTS_PER_SECOND))  # Resolution of LiDAR scan

        return lidar_bp

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

    def setup_lidar(self):

        # LiDAR
        lidar_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        self.lidar_sensor = self.world.spawn_actor(self.lidar_blueprint('sensor.lidar.ray_cast'), lidar_transform,
                                                   attach_to=self.car)
        weak_self = weakref.ref(self)
        self.lidar_sensor.listen(lambda lidar_data: weak_self().set_lidar(weak_self, lidar_data))

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

        # Segmentation camera
        seg_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera_segmentation = self.world.spawn_actor(self.camera_blueprint('sensor.camera.semantic_segmentation'),
                                                          seg_transform, attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera_segmentation.listen(lambda image_seg: weak_self().set_segmentation(weak_self, image_seg))

        # RGB camera
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.8), carla.Rotation(pitch=-15))
        camera_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.camera = self.world.spawn_actor(self.camera_blueprint('sensor.camera.rgb'), camera_transform,
                                             attach_to=self.car)
        weak_self = weakref.ref(self)
        self.camera.listen(lambda image: weak_self().set_image(weak_self, image))

        # Radar
        # radar_transform = carla.Transform(carla.Location(x=1.6, z=1.7))
        # self.camera_radar = self.world.spawn_actor(self.camera_blueprint('sensor.other.radar'), radar_transform, attach_to=self.car)
        # weak_self = weakref.ref(self)
        # self.camera_radar.listen(lambda radar_data: weak_self().set_radar(weak_self, radar_data))

        # Calibration
        calibration = np.identity(3)
        calibration[0, 2] = VIEW_WIDTH / 2.0
        calibration[1, 2] = VIEW_HEIGHT / 2.0
        calibration[0, 0] = calibration[1, 1] = VIEW_WIDTH / (2.0 * np.tan(VIEW_FOV * np.pi / 360.0))
        self.camera.calibration = calibration
        self.camera_segmentation.calibration = calibration

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
        if keys[K_c]:
            self.screen_capture = self.screen_capture + 1
        else:
            self.screen_capture = 0

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
            cv2.imwrite('custom_data/image' + str(self.image_count) + '.png', i3)
            print("RGB(custom)Image")

    @staticmethod
    def set_segmentation(weak_self, img):
        """
        Sets image coming from camera sensor.
        The self.capture flag is a mean of synchronization - once the flag is
        set, next coming image will be stored.
        """

        self = weak_self()
        if self.capture_segmentation:
            self.segmentation_image = img
            self.capture_segmentation = False

        if self.seg_record:
            img.convert(cc.CityScapesPalette)
            i = np.array(img.raw_data)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            i3 = i2[:, :, :3]
            cv2.imwrite('SegmentationImage/seg' + str(self.image_count) + '.png', i3)
            print("SegmentationImage")

    @staticmethod
    # def set_radar(weak_self, radar_data):
    #    self = weak_self()
    #    if self.capture_radar:
    #        self.radar_data = radar_data
    #        self.capture_radar = False

    #    if self.radar_record:
    #        # Extracting radar data and saving it to a file
    #        points = np.frombuffer(radar_data.raw_data, dtype=np.float32).reshape(-1, 4)
    #        # Save the radar data to a text file (azimuth, altitude, sensor and velocity)
    #        np.savetxt('RadarData/radar' + str(self.image_count) + '.txt', points)
    #        print("RadarData")

    @staticmethod
    def set_lidar(weak_self, lidar_data):
        self = weak_self()
        if self.capture_lidar:
            self.lidar_data = lidar_data
            self.capture_lidar = False

        if self.lidar_record:
            # Extract LiDAR points and save to a file
            points = np.frombuffer(lidar_data.raw_data, dtype=np.float32).reshape(-1, 4)
            # Save the LiDAR points to a text file (xyz coordinates, intensity loss)
            np.savetxt('LiDARData/lidar' + str(self.image_count) + '.txt', points)
            print("LiDARData")

    @staticmethod
    def set_depth(weak_self, depth_img):
        self = weak_self()
        if self.capture_depth:
            self.depth_image = depth_img
            self.capture_depth = False

        if self.depth_record:
            i = np.array(depth_img.raw_data, dtype=np.uint8)
            i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
            bgra_image_array = i2[:, :, :3]
            # Select a pixel at (x, y) - replace x and y with specific coordinates
            x, y = VIEW_WIDTH // 2, VIEW_HEIGHT // 2  # example coordinates
            pixel_value = bgra_image_array[y, x]

            # Convert pixel color values to distance
            B, G, R = pixel_value  # Assuming bgra_image_array is in BGRA format
            normalized = (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1)
            in_meters = 1000 * normalized
            return in_meters
            print(f"Pixel value at ({x}, {y}): {pixel_value}")
            print(f"Distance at ({x}, {y}): {in_meters} meters")

    def test_NN():

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
            vehicles = self.world.get_actors().filter('vehicle.*')
            pedestrians = self.world.get_actors().filter('walker.pedestrian.*')

            self.image_count = 0
            self.time_interval = 0

            global vehicle_bbox_record
            global pedestrian_bbox_record
            global count

            while True:
                self.world.tick()

                self.capture = True
                pygame_clock.tick_busy_loop(60)

                self.render(self.display)

                self.time_interval += 1
                if ((self.time_interval % args.CaptureLoop) == 0 and self.loop_state):
                    self.image_count = self.image_count + 1
                    self.rgb_record = True
                    self.seg_record = True
                    # self.radar_record = True
                    # self.lidar_record = True
                    self.depth_record = True
                    vehicle_bbox_record = True
                    pedestrian_bbox_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("ImageCount - %d" % self.image_count)

                if self.screen_capture == 1:
                    self.image_count = self.image_count + 1
                    self.rgb_record = True
                    self.seg_record = True
                    # self.radar_record = True
                    # self.lidar_record = True
                    self.depth_record = True
                    vehicle_bbox_record = True
                    pedestrian_bbox_record = True
                    count = self.image_count
                    print("-------------------------------------------------")
                    print("Captured! ImageCount - %d" % self.image_count)

                bounding_boxes = VehicleBoundingBoxes.get_bounding_boxes(vehicles, self.camera)
                pedestrian_bounding_boxes = PedestrianBoundingBoxes.get_bounding_boxes(pedestrians, self.camera)

                VehicleBoundingBoxes.draw_bounding_boxes(self.display, bounding_boxes)
                PedestrianBoundingBoxes.draw_bounding_boxes(self.display, pedestrian_bounding_boxes)

                time.sleep(0.03)
                self.rgb_record = False
                self.seg_record = False
                # self.radar_record = False
                # self.lidar_record = False
                self.depth_record = False
                pygame.display.flip()

                pygame.event.pump()
                if self.control(self.car):
                    return

        finally:
            self.set_synchronous_mode(False)
            self.camera.destroy()
            self.camera_segmentation.destroy()
            # self.camera_radar.destroy()
            # self.lidar_sensor.destroy()
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
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '-l', '--CaptureLoop',
        metavar='N',
        default=100,
        type=int,
        help='set Capture Cycle settings, Recommand : above 100')

    args = argparser.parse_args()

    print(__doc__)

    try:
        client = BasicSynchronousClient()
        client.game_loop(args)
    finally:
        print('EXIT')


if __name__ == '__main__':
    main()
