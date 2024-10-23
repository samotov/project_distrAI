import carla
import math
import random
import time

# Connect to world
client = carla.Client('localhost', 2000)    # Connect to the simulator, 2000 is default port
world = client.get_world()                  # Contains everything (vehicles, pedestrians, roads, ...)
bp_lib = world.get_blueprint_library()      # Contains blueprints of all objects in world

for vehc in world.get_actors().filter("*vehicle*"):
    vehc.destroy()
