import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torch.utils.data import Dataset

class BoundingBoxRegressionDataset(Dataset):

    def __init__(self, input_folder, transform = None, train = True):
        self.input_folder = input_folder
        self.transform = transform
        self.data = list()
        self.image_boundingbox_data = list()

        self.compose_data_lists(train)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        features, result = self.data[index]

        if self.transform:
            features = self.transform(features)

        return features, result

    def compose_data_lists(self, train):
        # We get a list of all the feature files and assume that the corresponding result file has the same name.
        subfolder = 'train' if train else 'val'
        feature_dir = os.path.join(self.input_folder, 'labels', subfolder)
        file_names = os.listdir(feature_dir)

        # We loop over all the filenames
        for file_name in file_names:
            feautures_path = os.path.join(self.input_folder, 'labels', subfolder, file_name)
            results_path = os.path.join(self.input_folder, '3D_information', subfolder, file_name)
            image_dir = os.path.join(self.input_folder, 'images', subfolder, file_name.replace('.txt', '.png'))

            # We open the files
            feature_file = open(feautures_path, 'r')
            results_file = open(results_path, 'r')

            # We read the lines of the files
            feature_lines = feature_file.readlines()
            results_lines = results_file.readlines()

            # We collect the results for each image
            results = list()

            # We go over each line
            for line_index in range(len(feature_lines)):
                # We get the feature and results line
                feature_line = feature_lines[line_index]
                result_line = results_lines[line_index]

                # We extract the features and the result and put them in a tensor
                features = torch.tensor([float(feature) for feature in feature_line.split()])
                result = torch.tensor([float(result_value) for result_value in result_line.split()])

                # We add the data to the data list
                self.data.append([features, result])
                results.append(result)

            # We save the image and results in another list to visualize them together
            self.image_boundingbox_data.append([image_dir, results]) 

            # we close the files
            feature_file.close()
            results_file.close()
    
    def visualize_data(self, amount):
        for i in range(amount):
            # We collect the data
            image_dir, results = self.image_boundingbox_data[i]

            # We start plotting the figure and plot the image in the first subplot
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            image = plt.imread(image_dir)
            ax1.imshow(image)
            ax1.axis('off')
            
            # We plot the sacond subplot, the 3D recreation of the image
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')

            # We initialize the transformation matrix to transform the boundingbox axis
            axis_transformation = np.array([
                [1, 0, 0],
                [0, 0, 1],
                [0, -1, 0]])

            # We loop over the results and plot their 3D boundingbox
            for result in results:
                x, y, z, h, w, l, _, rotation_y = result

                # We initialize the transformation matrix to ratate over rotation_y
                rotation_y_transformation = np.array([
                    [np.cos(rotation_y), 0, np.sin(rotation_y)],
                    [0, 1, 0],
                    [-np.sin(rotation_y), 0, np.cos(rotation_y)]])
            
                self.draw_3d_bounding_box((x, y, z), (h, w, l), ax2, rotation_y_transformation, axis_transformation)

            # Set labels and plot limits
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim([-20, 20])
            ax2.set_ylim([-20, 20])
            ax2.set_zlim([-20, 20])
            plt.show()

    def draw_3d_bounding_box(self, location, size, ax, trans_rot, trans_ax):
        # We get the location (x, y, z) of the objects center x = to the right, y = downward from the camera, z = into the scene
        x_center, y_center, z_center = location

        # We get the size (height, width, length)
        h, w, l = size

        # We calculate the corner coordinates
        x1 = x_center - (w/2)
        y1 = y_center - (h/2)
        z1 = z_center - (l/2)
        x2 = x_center + (w/2)
        y2 = y_center + (h/2)
        z2 = z_center + (l/2)
    
        # We define the corner points of the bounding box
        points = [
            [x1, y1, z1], [x2, y1, z1], [x2, y2, z1], [x1, y2, z1],  # Bottom square
            [x1, y1, z2], [x2, y1, z2], [x2, y2, z2], [x1, y2, z2]   # Top square
        ]

        points = [trans_ax @ (trans_rot @ np.array(point)) for point in points]
    
        # We define the edges by specifying pairs of points that should be connected
        edges = [
            [points[0], points[1]], [points[1], points[2]], [points[2], points[3]], [points[3], points[0]],  # Bottom square
            [points[4], points[5]], [points[5], points[6]], [points[6], points[7]], [points[7], points[4]],  # Top square
            [points[0], points[4]], [points[1], points[5]], [points[2], points[6]], [points[3], points[7]]   # Vertical lines
        ]
    
        # Plot each edge
        for edge in edges:
            xs, ys, zs = zip(*edge)
            ax.plot(xs, ys, zs, color='b')




