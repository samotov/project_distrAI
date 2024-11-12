import os
import yaml

class DataConverter:
    def __init__(self, input_dir, output_dir, classes):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.classes = classes

    def create_data_folders(self):
        # We define the subfolder structure
        folders = [
            'images/train',
            'images/val',
            'labels/train',
            'labels/val']

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

    def create_yaml_file(self):
        yaml_path = self.output_dir + ".yaml"
        # We define the content for the YAML file

        data = {
            'train': os.path.join(self.output_dir, "images", "train"),    # Path to the training data
            'val': os.path.join(self.output_dir, "images", "val"),        # Path to the validation data
            'nc': len(self.classes),        # Number of classes
            'names': list(self.classes)     # List of class names
        }

        # We write the data to the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)
        print(f"YAML file created: {yaml_path}")
