from .config import Config
import carla
import numpy as np

class RGBSensor():
    def __init__(self):
        self.rgb_camera = None
        self.rgb_record = True
        self.config = Config()
    
    def activate():
        self.rgb_record = True

    def deactivate():
        self.rgb_record = False
        
    def set_rgb(self, rgb_image):
        i = np.array(rgb_image.raw_data)
        i2 = i.reshape((self.config.view_height, self.config.view_width, 4))
        i3 = i2[:, :, :3]
        
        return i3
    
    def capture_image(self, rgb_image):
        if self.rgb_record:
            return self.set_rgb(rgb_image)

    def setup_rgb_camera(self, rgb_bp, vehicle):
        # Set rgb sensor-specific attributes using symbolic parameters
        rgb_bp.set_attribute('image_size_x', str(self.config.view_width))    # Width of the output rgb image
        rgb_bp.set_attribute('image_size_y', str(self.config.view_height))    # Height of the output rgb image
        rgb_bp.set_attribute('fov', str(self.config.view_fov))                       # Field of view of the rgb sensor
        # rgb_bp.set_attribute('sensor_tick', str(self.config.sensor_tick))       # Time interval for sensor updates
        
        # Set calibration of rgb camera
        calibration = np.identity(3)
        calibration[0, 2] = self.config.view_width / 2.0
        calibration[1, 2] = self.config.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.config.view_width / (2.0 * np.tan(self.config.view_fov * np.pi / 360.0))
        
        return rgb_bp, calibration