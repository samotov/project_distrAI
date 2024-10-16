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

import carla
import random
import time
import numpy as np
import cv2


def process_img(image, path):
    image.save_to_disk(path)



actor_list = []
try:
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)

    # Connect to world
    client = carla.Client('localhost', 2000)    # Connect to the simulator, 2000 is default port
    world = client.get_wordl()                  # Contains everything (vehicles, pedestrians, roads, ...)

    bp_lib = world.get_blueprint_library()      # Contains blueprints of all objects in world
    bp = blueprint_library.filter('model3')[0]

    spawn_point = random.choice(world.get_map().get_spawn_points())
    
    vehicle = world.spawn_actor(bp, spawn_point)
    vehicle.apply_control(carla.VehicleControl(throttle=1.0, steer=0.0))
    actor_list.append(vehicle)

    camera_bp = bp_lib.find("censor.camera.rgb")

    # Adjust sensor relative to vehicle
    spawn_point_camera = carla.Transform(carla.Location(x=2.5, z=0.7))
    spawn_point_depth = spawn_point_camera

    camera_path = "./camera_pictures/"
    depth_camera_path = "./depth_camera_pictures/"
    
    # Set up camera
    camera = world.spawn_actor(camera_bp, spawn_point_camera, attach_to=vehicle)                # spawn the sensor and attach to vehicle.
    actor_list.append(camera)                                                                   # add sensor to list of actors
    camera.listen(lambda data: process_img(data, camera_path + "%06d" % image.frame))                                  # do something with this sensor

    # Set up depth
    depth_camera = world.spawn_actor(depth_camera_bp, spawn_point_depth_camera, attach_to=vehicle)          # spawn the sensor and attach to vehicle.
    actor_list.append(depth_camera)                                                                         # add sensor to list of actors
    depth_camera.listen(lambda data: process_img(data, depth_camera_depth + camera_path = "%06d" % image.frame))                                 # do something with this sensor

    # sleep for 5 seconds, then finish:
    time.sleep(5)

finally:

    # Need to destroy actors, otherwise they will stay active after shutting down client-server connection
    print('destroying actors')
    for actor in actor_list:
        actor.destroy()
    print('done.')