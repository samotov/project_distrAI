from data_converters import DataConverter
import os
import progressbar

# Information about the dataset https://medium.com/@abdulhaq.ah/explain-label-file-of-kitti-dataset-738528de36f4

class KittiDataConverter(DataConverter.DataConverter):
    def __init__(self, input_dir, output_dir, classes, dataset):
        super().__init__(output_dir, classes)
        self.dataset = dataset
        self.input_dir = input_dir

    def convert_data(self, training_percentage):
        # We create the needed files and folders
        self.create_yaml_file()
        self.create_data_folders_kitti()

        # We get the amount of images
        num_training_images = int(len(self.dataset)*training_percentage)

        # We add a progressbar to see the progress of the conversion
        widgets = [progressbar.Percentage (), progressbar.Bar (), progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.dataset))
        bar.start()

        for image_index in range(len(self.dataset)):
            # We get the image and information about it
            image, information = self.dataset.__getitem__(image_index)
            width, heigth = image.size

            # We define wether we want this image to be a train image or a validation image and define the paths accordingly
            mode = 'train' if image_index < num_training_images else 'val'
            image_output_path = os.path.join('datasets', self.output_dir, 'images', mode, str(image_index) + '.png')
            labels_ouput_path = os.path.join('datasets', self.output_dir, 'labels', mode, str(image_index) + '.txt')
            information_3D_output_path = os.path.join('datasets', self.output_dir, '3D_information', mode, str(image_index) + '.txt')

            # We save the image in the output path
            image.save(image_output_path)

            # Now we will create a file with the bbox information and 3D information
            with open(labels_ouput_path, 'w') as file:
                for object in information:
                    #The type of the object
                    type = object['type']

                    #If we are intrested in this object type we procces it
                    if type in self.classes:
                        # We ge the class number
                        class_number = self.classes.index(type)

                        x1_2D, y1_2D, x2_2D, y2_2D = object['bbox']

                        # We normalize and clip the values and calculate the center, width and height of the boundingbox
                        x1_2D_clip_norm = max(0, min(1, x1_2D/width))
                        y1_2D_clip_norm = max(0, min(1, y1_2D/heigth))
                        x2_2D_clip_norm = max(0, min(1, x2_2D/width))
                        y2_2D_clip_norm = max(0, min(1, y2_2D/heigth))

                        x_center = (x1_2D_clip_norm + x2_2D_clip_norm)/2
                        y_center = (y1_2D_clip_norm + y2_2D_clip_norm)/2
                        w = x2_2D_clip_norm - x1_2D_clip_norm
                        h = y2_2D_clip_norm - y1_2D_clip_norm

                        # We write the information in a YOLO trainable format
                        # We define the string and write it to the .txt file
                        bbox_string = str(class_number) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h)
                        file.write(bbox_string + '\n')

            with open(information_3D_output_path, 'w') as file:
                for object in information:
                    #The type of the object
                    type = object['type']

                    #If we are intrested in this object type we procces it
                    if type in self.classes:
                        # The 3D information in meters
                        x_3D, y_3D, z_3D = object['location']
                        heigth_3D, width_3D, length_3D = object['dimensions']
                        rotation_y = object['rotation_y']

                        information_3D_string = str(x_3D) + ' ' + str(y_3D) + ' ' + str(z_3D) + ' ' + str(heigth_3D) + ' ' + str(width_3D) + ' ' + str(length_3D) + ' ' + str(rotation_y)
                        file.write(information_3D_string + '\n')
            
            bar.update(image_index + 1)
        bar.finish()

    def create_data_folders_kitti(self):
        # We create the original folders and also create folders for the 3D information for later
        self.create_data_folders()
        folders = [
            '3D_information/train',
            '3D_information/val']

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


