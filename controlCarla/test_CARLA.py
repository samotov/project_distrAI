#https://www.youtube.com/watch?v=pONr1R1dy88&ab_channel=CARLASimulator

import carla
import math
import random
import time

# Connect to world
client = carla.Client('localhost', 2000)    # Connect to the simulator, 2000 is default port
world = client.get_wordl()                  # Contains everything (vehicles, pedestrians, roads, ...)
bp_lib = world.get_blueprint_library()      # Contains blueprints of all objects in world

# Spawn a vehicle
spawn_points = world.get_map().get_spawn_points() # Get the spawn points of the current map
vehicle_bp = bp_lib.find("vehicle.lincoln.mkz_2020")    # Get a specific vehicle
vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))    # TRY and spawn the vehicle at one of the predetermined 

# Activate spectator mode to fly around the world
spectator = world.get_spectator()
vehc_transform = carla.Transform(vehicle.get_transform(carla.Location(x=-4, z=2.5)), vehicle.get_transformation())
spectator.set_transform(vehc_transform)

# Add more vehicles
num_vehc = 20
for i in range(num_vehc):
    vehc_bp = random.choice(bp_lib.filter("vehicle")) # Get random vehicle
    npc = world.try_spawn_actor(vehc_bp, random.choice(spawn_points))

# Set vehicles in autopilot mode
for vehc in world.get_actors().filter("*vehicle*"):
    vehc.set_autopilot(True)

# Create camera
camera_bp = bp_lib.find("sensor.camera.rgb")
camera_init_trans = carla.Transform(carla.Location(z=2))
camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)

# Save images 
camera.listen(lambda image: image.save_to_disk("out/%06d" % image.frame))