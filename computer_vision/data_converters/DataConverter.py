import os
import yaml

# Parent class for all the data converters
class DataConverter:
    def __init__(self, output_dir, classes):
        self.output_dir = output_dir
        self.classes = classes

    # Creates the ouput folders in the YOLO format
    def create_data_folders(self):
        # We define the subfolder structure
        folders = [
            'images\\train',
            'images\\val',
            'labels\\train',
            'labels\\val']

        # We loop through each folder and create it
        for folder in folders:
            # Create the full path
            folder_path = os.path.join('datasets', self.output_dir, folder)

            # Check if the folder already exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")
            else:
                print(f"Folder already exists: {folder_path}")

    # Creates the yaml file of the dataset in the YOLO format
    def create_yaml_file(self):
        yaml_path = self.output_dir + ".yaml"
        # We define the content for the YAML file

        data = {
            'train': self.output_dir + "\\images\\train",    # Path to the training data
            'val': self.output_dir + "\\images\\val",        # Path to the validation data
            'nc': len(self.classes),        # Number of classes
            'names': list(self.classes)     # List of class names
        }

        # We write the data to the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)
        print(f"YAML file created: {yaml_path}")
