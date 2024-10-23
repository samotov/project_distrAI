import carla
import random
import pygame
import datetime
import numpy as np
import cv2
import csv
from PIL import Image 
import PIL

client = carla.Client('localhost',2000)
world = client.get_world()
weather = carla.WeatherParameters(
    cloudiness=0.0,
    precipitation=0.0,
    sun_altitude_angle=10.0,
    sun_azimuth_angle = 70.0,
    precipitation_deposits = 0.0,
    wind_intensity = 0.0,
    fog_density = 0.0,
    wetness = 0.0, 
)
world.set_weather(weather)

bp_lib = world.get_blueprint_library() 
spawn_points = world.get_map().get_spawn_points()

vehicle_bp = bp_lib.find('vehicle.audi.etron')
ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_points[79])

spectator = world.get_spectator()
transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),ego_vehicle.get_transform().rotation)
spectator.set_transform(transform)

for i in range(20):  
    vehicle_bp = random.choice(bp_lib.filter('vehicle')) 
    npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

for v in world.get_actors().filter('*vehicle*'): 
    v.set_autopilot(True) 
ego_vehicle.set_autopilot(False) 


# Add RGB camera
camera_bp = bp_lib.find('sensor.camera.rgb') 
camera_init_trans = carla.Transform(carla.Location(x =-0.1,z=1.7)) 
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle)

# Storing image width and height values
image_w = camera_bp.get_attribute("image_size_x").as_int()
image_h = camera_bp.get_attribute("image_size_y").as_int()

# Add depth camera
depth_camera_bp = bp_lib.find('sensor.camera.depth') 
depth_camera = world.spawn_actor(depth_camera_bp, camera_init_trans, attach_to=ego_vehicle)


# Callback functions for all the sensors used here.
def rgb_callback(image, data_dict):
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4)) #Reshaping with alpha channel
    img[:,:,3] = 255
    data_dict['rgb_image'] = img

def depth_callback(image, data_dict):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    data_dict['depth_image'] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

# Update the sensor_data dictionary to include the depth_image, gnss and imu keys and default values
sensor_data = {'rgb_image': np.zeros((image_h, image_w, 4)),
               'depth_image': np.zeros((image_h, image_w, 4))}

# We'll create a simple function which will save the image with the timestamp
# as the name of the image

def rgb_image_creator(image,date_time):
#    cv2.imwrite(f'rgb_{date_time}.jpg',image)
#    pygame.image.save(image, f'rgb_{date_time}.jpg')
    image = image[:,:,0:3]
    im = Image.fromarray(image,"RGB")
    im.save(f"./rgb_images/rgb_{date_time}")
 
def depth_image_creator(image,date_time):
#    cv2.imwrite(f'depth_{date_time}.jpg',image)
#    pygame.image.save(image, f'depth{date_time}.jpg')
    im = Image.fromarray(image,"RGB")
    im.save(f"./depth_images/depth_{date_time}")

def keyboard_control(keys):

    if keys[pygame.K_UP] or keys[pygame.K_w]:
        control.throttle = min(control.throttle + 0.05, 1.0)
    else:
        control.throttle = 0.0

    if keys[pygame.K_DOWN] or keys[pygame.K_s]:
        control.brake = min(control.brake + 0.2, 1.0)
    else:
        control.brake = 0.0

    if keys[pygame.K_LEFT] or keys[pygame.K_a]:
        control.steer = max(control.steer - 0.05, -1.0)
    elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
        control.steer = min(control.steer + 0.05, 1.0)
    else:
        control.steer = 0.0

    control.hand_brake = keys[pygame.K_SPACE]
    # Apply the control to the ego vehicle 
    ego_vehicle.apply_control(control)

# Listen to the sensor feed
camera.listen(lambda image: rgb_callback(image, sensor_data))
depth_camera.listen(lambda image: depth_callback(image, sensor_data))

pygame.init() 

size = (640, 480)
pygame.display.set_caption("CARLA Manual Control")
screen = pygame.display.set_mode(size)

control = carla.VehicleControl()
clock = pygame.time.Clock()
done = False

while not done:

    keys = pygame.key.get_pressed() 
    
    # Made the keyboard control into a function
    keyboard_control(keys)

    # We'll use the datetime library to get the present time
    # We'll name our images with the corresponding timestamp
    
    current_time = datetime.datetime.now()
    date_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Let's now save the images as well
    rgb_image_creator(sensor_data['rgb_image'],datetime)
    depth_image_creator(sensor_data['depth_image'],datetime)

    # tick the simulation
    world.tick()

    # Update the display and check for the quit event
    pygame.display.flip()
    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    # Sleep to ensure consistent loop timing
    clock.tick(60)

for actor in world.get_actors():
    actor.destroy()