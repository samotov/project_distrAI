import os
import keyboard
import shutil
import cv2
import matplotlib.pyplot as plt
from data_converters import DataConverter

class YoloDatasetFilterer(DataConverter.DataConverter):
    def __init__(self, input_dir, output_dir, classes):
        super().__init__(input_dir, output_dir, classes)
        
        self.create_yaml_file()
        self.create_data_folders()

    def filter_data(self, start_index, image_refresh_rate):
        for mode in ['train', 'val']:
            # We define the data folders
            images_input_path = os.path.join('datasets', self.input_dir, 'images', mode)
            labels_input_path = os.path.join('datasets', self.input_dir, 'labels', mode)
            images_output_path = os.path.join('datasets', self.output_dir, 'images', mode)
            labels_output_path = os.path.join('datasets', self.output_dir, 'labels', mode)

            # We get the amount of data samples
            image_files = [f for f in os.listdir(images_input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            for image_index in range(start_index, len(image_files)):
                # We define the specific data folders
                image_input_path = os.path.join(images_input_path, str(image_index) + '.png')
                label_input_path = os.path.join(labels_input_path, str(image_index) + '.txt')
                image_output_path = os.path.join(images_output_path, str(image_index) + '.png')
                label_output_path = os.path.join(labels_output_path, str(image_index) + '.txt')

                # We read the image
                img = cv2.imread(image_input_path)
                height, width, _ = img.shape

                # We draw the boundingboxes and show the image
                with open(label_input_path, 'r') as file:
                    for line in file:
                        _ , x, y, w, h = [float(i) for i in line.split()]
                        x1 = int((x - (w/2)) * width)
                        x2 = int((y - (h/2)) * height)
                        y1 = int((x + (w/2)) * width)
                        y2 = int((y + (h/2)) * height)

                        # We draw the rectangle
                        cv2.rectangle(img, (x1, x2), (y1, y2), (255, 0, 0), 2)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.show(block = False)
                plt.pause(image_refresh_rate)
                plt.close()
                
                # We save the image and label data in the output folder in function of what key we press
                key = None
                while key not in ["enter", "backspace"]:
                     key = keyboard.read_key()
                
                # If enter is pressed we save the images otherwise (if backspace is pressed) we do not
                if key == "enter":
                    shutil.copy(image_input_path, image_output_path)
                    shutil.copy(label_input_path, label_output_path)
                    print('Moved '+ image_input_path + ' to ' + image_output_path)
                    print('Moved '+ label_input_path + ' to ' + label_output_path)