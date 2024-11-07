import numpy as np
import math

# Information about the axis of the different sensors:  https://www.cvlibs.net/publications/Geiger2013IJRR.pdf

class BoundingBoxTransformer:
    def __init__(self, x_center, y2, z_center, h, w, l, rotation_y):
        self.x_center = x_center
        self.y2 = y2
        self.z_center = z_center
        self.h = h
        self.w = w
        self.l = l
        self.rotation_y = rotation_y
        self.points = list()
        self.transform_coordinates()
    
    def transform_coordinates(self):
        # The translation to the origin
        t_origin = np.array([
            [1, 0, 0, -self.x_center],
            [0, 1, 0, 0],
            [0, 0, 1, -self.z_center],
            [0, 0, 0, 1]])
        
        # The rotation along the y_axis
        rot_angle = self.rotation_y - (math.pi/2)       # To orient the angle correctly we sutract pi/2
        t_rot = np.array([
            [np.cos(rot_angle), 0, np.sin(rot_angle), 0],
            [0, 1, 0, 0],
            [-np.sin(rot_angle), 0, np.cos(rot_angle), 0],
            [0, 0, 0, 1]])
        
        # The translation back  
        t_back = np.array([
            [1, 0, 0, self.x_center],
            [0, 1, 0, 0],
            [0, 0, 1, self.z_center],
            [0, 0, 0, 1]])
        
        # The axis transformation
        t_axis = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1]])

        # We calculate the corner coordinates
        x1 = self.x_center - (self.w/2)
        y1 = self.y2 - self.h
        z1 = self.z_center - (self.l/2)
        x2 = self.x_center + (self.w/2)
        y2 = self.y2
        z2 = self.z_center + (self.l/2)
    
        # We define the corner points of the bounding box
        points = [
            [x1, y1, z1, 1], [x2, y1, z1, 1], [x2, y2, z1, 1], [x1, y2, z1, 1],  # Bottom square
            [x1, y1, z2, 1], [x2, y1, z2, 1], [x2, y2, z2, 1], [x1, y2, z2, 1]   # Top square
        ]

        # We transform the points with our transformation matrices and remove the 1 and the end to transform them 
        # from homogenous coordinates to cartesian ones.
        self.points = [(t_axis @ (t_back @ (t_rot @ (t_origin @ np.array(point))))).tolist()[:-1] for point in points]

    def get_cornerpoints(self):
        return self.points[0], self.points[6]