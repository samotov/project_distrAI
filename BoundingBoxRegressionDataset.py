import os
import torch
from torch.utils.data import Dataset

class BoundingBoxRegressionDataset(Dataset):

    def __init__(self, input_folder, transform = None, train = True):
        self.input_folder = input_folder
        self.transform = transform
        self.data = list()

        self.compose_data_dictionary(train)

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        features, result = self.data[index]

        if self.transform:
            features = self.transform(features)

        return features, result

    def compose_data_dictionary(self, train):
        # We get a list of all the feature files and assume that the corresponding result file has the same name.
        subfolder = 'train' if train else 'val'
        feature_dir = os.path.join(self.input_folder, 'labels', subfolder)
        file_names = os.listdir(feature_dir)

        # We loop over all the filenames
        for file_name in file_names:
            feautures_path = os.path.join(self.input_folder, 'labels', subfolder, file_name)
            results_path = os.path.join(self.input_folder, '3D_information', subfolder, file_name)

            # We open the files
            feature_file = open(feautures_path, 'r')
            results_file = open(results_path, 'r')

            # We read the lines of the files
            feature_lines = feature_file.readlines()
            results_lines = results_file.readlines()

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

            # we close the files
            feature_file.close()
            results_file.close()




