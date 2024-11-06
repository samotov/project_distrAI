# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles. The agent also responds to traffic lights.
It can also make use of the global route planner to follow a specifed route
"""

import carla

from queue import Queue
from agents.navigation.basic_agent import BasicAgent
from sensors import ObjectDetector, DepthSensor

class ACCAgent(BasicAgent):
    """
    ACCAgent implements an agent that navigates the scene at a fixed velocity.
    This agent will fail if asked to perform turns that are impossible are the desired speed.
    This includes lane changes. When a collision is detected, the constant velocity will stop,
    wait for a bit, and then start again.
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

        self._use_basic_behavior = False  # Whether or not to use the BasicAgent behavior when the constant velocity is down
        self._target_speed = target_speed / 3.6  # [m/s]
        self._current_speed = vehicle.get_velocity().length()  # [m/s]
        self._constant_velocity_stop_time = None
        self._collision_sensor = None
        self._object_detector = None
        self._depth_sensor = None
        self._rgb_sensor = None

        self._image_dictionary = dict() # Object that will hold all the information from the sensors

        self._rgb_sensor = self._setup_RGB_camera()
        self._depth_camera = self._setup_depth_camera()

        self._image_queue = Queue()

        self._restart_time = float('inf')  # Time after collision before the constant velocity behavior starts again

        if 'restart_time' in opt_dict:
            self._restart_time = opt_dict['restart_time']
        if 'use_basic_behavior' in opt_dict:
            self._use_basic_behavior = opt_dict['use_basic_behavior']

        self.is_constant_velocity_active = True
        self._set_collision_sensor()
        self._set_constant_velocity(target_speed)
        self._set_object_detector()

    def set_target_speed(self, speed):
        """Changes the target speed of the agent [km/h]"""
        self._target_speed = speed / 3.6
        self._local_planner.set_speed(speed)

    def stop_constant_velocity(self):
        """Stops the constant velocity behavior"""
        self.is_constant_velocity_active = False
        self._vehicle.disable_constant_velocity()
        self._constant_velocity_stop_time = self._world.get_snapshot().timestamp.elapsed_seconds

    def restart_constant_velocity(self):
        """Public method to restart the constant velocity"""
        self.is_constant_velocity_active = True
        self._set_constant_velocity(self._target_speed)

    def _set_constant_velocity(self, speed):
        """Forces the agent to drive at the specified speed"""
        self._vehicle.enable_constant_velocity(carla.Vector3D(speed, 0, 0))

    def _set_object_detector(self):
        """Instantiate the object detector"""
        self._object_detector = ObjectDetector()

    def _setup_RGB_camera(self):
        """Instantiate the RGB camera"""
        # Create rgb sensor class that will handle everything concerning rgb images
        self._rgb_sensor = RGBSensor()

        rgb_bp = self._world.get_blueprint_library.find("sensor.camera.rgb")
        self.rgb_camera = self._rgb_sensor.set up_rgb_camera(rgb_bp, self.vehicle)
        self.rgb_camera.listen(lambda rgb_image: self._rgb_sensor.capture_image(rgb_image, self._image_dictionary))

    
    def _setup_depth_camera(self):       
        """
        Returns depth blueprint with specified symbolic attributes.
        """
        # Create depth sensor class that will handle everything concerning depth images
        self._depth_sensor = DepthSensor()

        depth_bp = self._world.get_blueprint_library.find("sensor.camera.depth")
        self.depth_camera = self._depth_sensor.setup_depth_camera(depth_bp, self.vehicle)
        self.depth_camera.listen(lambda depth_image: self._depth_sensor.capture_image(depth_image, self._image_dictionary))
        
    def run_step(self):
        """Execute one step of navigation."""
        if not self.is_constant_velocity_active:
            if self._world.get_snapshot().timestamp.elapsed_seconds - self._constant_velocity_stop_time > self._restart_time:
                self.restart_constant_velocity()
                self.is_constant_velocity_active = True
            elif self._use_basic_behavior:
                return super(ACCAgent, self).run_step()
            else:
                return carla.VehicleControl()

        hazard_detected = False

        # Retrieve all relevant actors
        actor_list = self._world.get_actors()

        # Get current vehicle speed to calculate radius in which to check for cars/lights
        vehicle_speed = self._vehicle.get_velocity().length()

        # Check if the vehicle is affected by other vehicles. 
        # TODO: Use computer vision for this.
        # =====================================================================================================
        # Calculate the distance at which the vehicle should react        
        max_vehicle_distance = self._base_vehicle_threshold + vehicle_speed

        # Get RGB and depth image
        RGB_image = self._image_dictionary["rgb_image"]
        depth_image = self._image_dictionary["bgra_image"]
        depth_logarithmic = self._image_dictionary["grayscale_image"]

        # Get all the boundingboxes from vehicles in the area        
        vehicle_list = self._object_detector.extract_objects_from_surrounding(RGB_image, depth_image, depth_log_image)
        
        # Check if vehicle is in the range of concern
        affected_by_vehicle, vehicle, distance = self._vehicle_obstacle_detected(vehicle_list, max_vehicle_distance)
        if affected_by_vehicle:

            # TODO: use RL agent to determine velocity
            # ====================================================================================================
            vehicle_velocity = self._vehicle.get_velocity()
            # If the car is standing still, stay still
            if vehicle_velocity.length() == 0:
                hazard_speed = 100
            
            # If the car is not standing still, temper the velocity
            else:
                hazard_speed = vehicle_velocity.dot(adversary.get_velocity()) / vehicle_velocity.length()
            hazard_detected = True
        # ====================================================================================================

        # FUTURE TODO: use computer vision to check the traffic lights and react to it
        # ====================================================================================================        
        # Check if the vehicle is affected by a red traffic light
        lights_list = actor_list.filter("*traffic_light*")
        max_tlight_distance = self._base_tlight_threshold + 0.3 * vehicle_speed
        affected_by_tlight, _ = self._affected_by_traffic_light(lights_list, max_tlight_distance)
        if affected_by_tlight:
            hazard_speed = 100
            hazard_detected = True
        # ====================================================================================================

        # The longitudinal PID is overwritten by the constant velocity but it is
        # still useful to apply it so that the vehicle isn't moving with static wheels
        control = self._local_planner.run_step()
        
        # Change the speed
        if hazard_detected:
            self._set_constant_velocity(hazard_speed)
        else:
            self._set_constant_velocity(self._target_speed)

        return control

    def _set_collision_sensor(self):
        blueprint = self._world.get_blueprint_library().find('sensor.other.collision')
        self._collision_sensor = self._world.spawn_actor(blueprint, carla.Transform(), attach_to=self._vehicle)
        self._collision_sensor.listen(lambda event: self.stop_constant_velocity())

    def destroy_sensor(self):
        if self._collision_sensor:
            self._collision_sensor.destroy()
            self._collision_sensor = None

    def _vehicle_obstacle_detected(vehicle_list, max_vehicle_distance):
        """
        Check if there are vehicles to be concerned about
        """
        return False, None, 0