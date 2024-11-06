# ====================================================================================
# Class that converts depth image to RGB and Grayscale depth images.
# Class based on methods from CarFree dataset creator
# 
# ====================================================================================
from .config import Config

class DepthSensor():
    def __init__(self)
        self.depth_camera = None
        self.depth_record = True
        self.config = Config()
    
    def activate():
        self.depth_record = True

    def deactivate()
        self.depth_record = False
        
    @staticmethod
    def set_depth(depth_image):
        i = np.array(depth_image.raw_data)
        i2 = i.reshape((self.config.view_height, self.config.view_width, 4))
        bgra_image = i2[:, :, :3]  
        grayscale_image =  depth_to_logarithmic_grayscale(bgra_image)
        return bgra_image, grayscale_image
    
    @staticmethod
    def capture_image(depth_image, dictionary):
        if self.depth_record:
            bgra_image, grayscale_image = self.set_depth(depth_image)
            dictonary["bgra_image"] = bgra_image
            dictonary["grayscale_image"] = grayscale_image
    
    def setup_depth_camera(self, depth_bp, vehicle):
        # Set depth sensor-specific attributes using symbolic parameters
        depth_bp.set_attribute('image_size_x', str(self.config.view_width))    # Width of the output depth image
        depth_bp.set_attribute('image_size_y', str(self.config.view_height))    # Height of the output depth image
        depth_bp.set_attribute('fov', str(self.config.view_fov))                       # Field of view of the depth sensor
        depth_bp.set_attribute('self.config.sensor_tick', str(self.config.sensor_tick))       # Time interval for sensor updates

        # Create depth camera
        depth_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.depth_camera = self.world.spawn_actor(self.depth_blueprint('sensor.camera.depth'), depth_transform, attach_to=vehicle)
        
        # Set calibration of depth camera
        calibration = np.identity(3)
        calibration[0, 2] = self.config.view_width / 2.0
        calibration[1, 2] = self.config.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.config.view_width / (2.0 * np.tan(self.config.view_fov * np.pi / 360.0))
        
        self.depth_camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        self.depth_camera.calibration = calibration

        return self.depth_camera

    @staticmethod
    def to_bgra_array(image):
    """Convert a CARLA raw image to a BGRA numpy array."""
    array = numpy.frombuffer(image.raw_data, dtype=numpy.dtype("uint8"))
    array = numpy.reshape(array, (image.height, image.width, 4))
    return array

    @staticmethod
    def depth_to_logarithmic_grayscale(image):
    """
    Convert an image containing CARLA encoded depth-map to a logarithmic
    grayscale image array.
    "max_depth" is used to omit the points that are far enough.
    """
    normalized_depth = depth_to_array(image)
    # Convert to logarithmic depth.
    logdepth = numpy.ones(normalized_depth.shape) + \
        (numpy.log(normalized_depth) / 5.70378)
    logdepth = numpy.clip(logdepth, 0.0, 1.0)
    logdepth *= 255.0
    # Expand to three colors.
    return numpy.repeat(logdepth[:, :, numpy.newaxis], 3, axis=2)