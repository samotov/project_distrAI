# ====================================================================================
# Class that converts depth image to RGB and Grayscale depth images.
# Class based on methods from CarFree dataset creator
# 
# ====================================================================================
from .config import Config
import numpy as np
from PIL import Image

class DepthSensor():
    def __init__(self):
        self.depth_camera = None
        self.depth_record = True
        self.config = Config()
    
    def activate():
        self.depth_record = True

    def deactivate():
        self.depth_record = False

    def setup_depth_camera(self, depth_bp, vehicle):
        # Set depth sensor-specific attributes using symbolic parameters
        depth_bp.set_attribute('image_size_x', str(self.config.view_width))    # Width of the output depth image
        depth_bp.set_attribute('image_size_y', str(self.config.view_height))    # Height of the output depth image
        depth_bp.set_attribute('fov', str(self.config.view_fov))                       # Field of view of the depth sensor
        # depth_bp.set_attribute('sensor_tick', str(self.config.sensor_tick))       # Time interval for sensor updates

        # Set calibration of depth camera
        calibration = np.identity(3)
        calibration[0, 2] = self.config.view_width / 2.0
        calibration[1, 2] = self.config.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.config.view_width / (2.0 * np.tan(self.config.view_fov * np.pi / 360.0))

        return depth_bp, calibration
        
    def capture_image(self, depth_image):
        if self.depth_record:
            return self.set_depth(depth_image)
   
        
    def set_depth(self, depth_image):
        """Extract bgra and grayscale image from received depth image"""
        i = np.array(depth_image.raw_data)                                      # Convert image data into array
        i2 = i.reshape((self.config.view_height, self.config.view_width, 4))    # Reshape the data into rgba matrix (tensor)
        bgr_image = i2[:, :, :3]                                                # Extract only BGR information
        grayscale_image = self.depth_to_logarithmic_grayscale(bgr_image)        # Extract logarithmic grayscale from BGR information
        return bgr_image, grayscale_image
   
    @staticmethod
    def to_bgra_array(image):
        """Convert a CARLA raw image to a BGRA np array."""
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        return array

    def depth_to_array(self, image_array):
        """
        Convert an image containing CARLA encoded depth-map to a 2D array containing
        the depth value of each pixel normalized between [0.0, 1.0].
        """
        # Apply (R + G * 256 + B * 256 * 256) / (256 * 256 * 256 - 1).
        normalized_depth = np.dot(image_array[:, :, :3], [65536.0, 256.0, 1.0])
        normalized_depth /= 16777215.0  # (256.0 * 256.0 * 256.0 - 1.0)
        return normalized_depth

    def depth_to_logarithmic_grayscale(self, image):
        """
        Convert an image containing CARLA encoded depth-map to a logarithmic
        grayscale image array.
        "max_depth" is used to omit the points that are far enough.
        """
        normalized_depth = self.depth_to_array(image)
        # Convert to logarithmic depth.
        logdepth = np.ones(normalized_depth.shape) + \
            (np.log(normalized_depth) / 5.70378)
        logdepth = np.clip(logdepth, 0.0, 1.0)
        logdepth *= 255.0
        # Expand to three colors.
        return np.repeat(logdepth[:, :, np.newaxis], 3, axis=2)