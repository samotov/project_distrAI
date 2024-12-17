import torch

class BoundingBox3D:

    def __init__(self, point1, point2):
        self.x1, self.y1, self.z1 = point1
        self.x2, self.y2, self.z2 = point2
        self.points = list()
        self.calculate_points()
    
    def to_tensor(self):
        return torch.tensor([self.x1, self.y1, self.z1, self.x2, self.y2, self.z2])
    
    def calculate_points(self):
        self.points = [
            [self.x1, self.y1, self.z1], [self.x2, self.y1, self.z1], [self.x2, self.y2, self.z1], [self.x1, self.y2, self.z1],  # Bottom square
            [self.x1, self.y1, self.z2], [self.x2, self.y1, self.z2], [self.x2, self.y2, self.z2], [self.x1, self.y2, self.z2]   # Top square
        ]

    def get_cornerpoints(self):
        return [self.points[0], self.points[6]]
    
    def get_centerpoint_tensor(self):
        x_center = (self.x1 + self.x2)/2
        y_center = (self.y1 + self.y2)/2
        z_center = (self.z1 + self.z2)/2
        return torch.tensor([x_center, y_center, z_center])

    def get_x_y_centerpoints_tensor(self):
        x_center = (self.x1 + self.x2)/2
        y_center = (self.y1 + self.y2)/2
        return torch.tensor([x_center, y_center])

    def visualize_boundingbox(self, ax, color):
        # We define the edges by specifying pairs of points that should be connected
        edges = [
            [self.points[0], self.points[1]], [self.points[1], self.points[2]], [self.points[2], self.points[3]], [self.points[3], self.points[0]],  # Bottom square
            [self.points[4], self.points[5]], [self.points[5], self.points[6]], [self.points[6], self.points[7]], [self.points[7], self.points[4]],  # Top square
            [self.points[0], self.points[4]], [self.points[1], self.points[5]], [self.points[2], self.points[6]], [self.points[3], self.points[7]]   # Vertical lines
        ]
    
        # Plot each edge
        for edge in edges:
            xs, ys, zs = zip(*edge)
            ax.plot(xs, ys, zs, color=color)