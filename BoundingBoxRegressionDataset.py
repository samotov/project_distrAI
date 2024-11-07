import os
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from Boundingbox3D import BoundingBox3D
import BoundingBoxTransformer

# 0: 'Car', 1: 'Pedestrian', 2: 'Van', 3: 'Cyclist', 4: 'Truck', 5: 'Misc', 6: 'Tram', 7: 'Dontcare'

class BoundingBoxRegressionDataset(Dataset):

    def __init__(self, input_folder, transform = None, train = True):
        self.input_folder = input_folder
        self.transform = transform

        # We use these estimations for each class their bouningbox [h, w, l]
        self.estimated_boundingbox_sizes = {0: [1.5, 1.6, 4.2],     # car
                                            1: [1.85, 0.4, 0.4],    # pedestrian
                                            2: [2, 2.2, 5.3],       # van
                                            3: [1.7, 0.5, 1.95],    # cyclist
                                            4: [2.85, 2.65, 12.35], # truck
                                            5: [1.3, 0.8, 1.95],    # misc
                                            6: [3.6, 2.7, 35.24],   # tram
                                            7: [0, 0, 0]}           # don't care
        
        # One hot encoding of each class that the YoLo model detects
        self.classes_one_hot_encoding = {0: [1, 0, 0, 0, 0, 0, 0],  # car
                                         1: [0, 1, 0, 0, 0, 0, 0],  # pedestrian
                                         2: [0, 0, 1, 0, 0, 0, 0],  # van
                                         3: [0, 0, 0, 1, 0, 0, 0],  # cyclist
                                         4: [0, 0, 0, 0, 1, 0, 0],  # truck
                                         5: [0, 0, 0, 0, 0, 1, 0],  # misc
                                         6: [0, 0, 0, 0, 0, 0, 1],  # tram
                                         7: [0, 0, 0, 0, 0, 0, 0]}  # don't care
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
                features = [float(feature) for feature in feature_line.split()]
                x, y, z, h, w, l, rot_y = [float(result_value) for result_value in result_line.split()]

                # We replace the class index with the one hot encoding of the class to get the total features
                total_features = self.classes_one_hot_encoding[features[0]] + features[1:]

                # We transform the bounding box to the other coordinate system
                boundingbox_transformer = BoundingBoxTransformer.BoundingBoxTransformer(x, y, z, h, w, l, rot_y)
                point1, point2 = boundingbox_transformer.get_cornerpoints()
                result = BoundingBox3D(point1, point2).get_x_y_centerpoints_tensor()

                # We add the data to the data list
                self.data.append([torch.tensor(total_features), result])
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

            # We loop over the results and plot their 3D boundingbox
            for result in results:
                result.visualize_boundingbox(ax2, 'b')

            # Set labels and plot limits
            ax2.set_xlabel('X')
            ax2.set_ylabel('Y')
            ax2.set_zlabel('Z')
            ax2.set_xlim([-20, 20])
            ax2.set_ylim([-20, 20])
            ax2.set_zlim([-20, 20])
            plt.show()



