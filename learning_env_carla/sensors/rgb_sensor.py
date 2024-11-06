from .config import Config

class RGBSensor():
    def __init__(self)
        self.rgb_camera = None
        self.rgb_record = True
        self.config = Config()
    
    def activate():
        self.rgb_record = True

    def deactivate()
        self.rgb_record = False
        
    @staticmethod
    def set_rgb(rgb_image):
        i = np.array(img.raw_data)
        i2 = i.reshape((VIEW_HEIGHT, VIEW_WIDTH, 4))
        i3 = i2[:, :, :3]
        
        return i3
    
    @staticmethod
    def capture_image(rgb_image, dictionary):
        if self.rgb_record:
            rgb_image = self.set_rgb(rgb_image)
            dictonary["rgb_image"] = rgb_image
    
    def setup_rgb_camera(self, rgb_bp, vehicle):
        # Set rgb sensor-specific attributes using symbolic parameters
        rgb_bp.set_attribute('image_size_x', str(self.config.view_width))    # Width of the output rgb image
        rgb_bp.set_attribute('image_size_y', str(self.config.view_height))    # Height of the output rgb image
        rgb_bp.set_attribute('fov', str(self.config.view_fov))                       # Field of view of the rgb sensor
        rgb_bp.set_attribute('self.config.sensor_tick', str(self.config.sensor_tick))       # Time interval for sensor updates

        # Create rgb camera
        rgb_transform = carla.Transform(carla.Location(x=1.6, z=1.7), carla.Rotation(pitch=-15))
        self.rgb_camera = self.world.spawn_actor(self.rgb_blueprint('sensor.camera.rgb'), rgb_transform, attach_to=vehicle)
        
        # Set calibration of rgb camera
        calibration = np.identity(3)
        calibration[0, 2] = self.config.view_width / 2.0
        calibration[1, 2] = self.config.view_height / 2.0
        calibration[0, 0] = calibration[1, 1] = self.config.view_width / (2.0 * np.tan(self.config.view_fov * np.pi / 360.0))
        
        self.rgb_camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
        self.rgb_camera.calibration = calibration

        return self.rgb_camera