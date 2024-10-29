import cv2
import os
import numpy as np
import yaml

class DataConverter:
    def __init__(self, input_dir, output_dir, class_color_info_map):
        self.output_dir = output_dir
        self.input_dir = input_dir
        self.class_color_info_map = class_color_info_map


    def get_2D_bounding_box_from_segmeted_image(self, segmented_image_path):
        # We load the image
        img = cv2.imread(segmented_image_path)

        # We convert the image from BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Loop through each segmented color that we are intrested in
        boundingboxes_per_class = dict()
        for label in self.class_color_info_map.keys():
            color, dilation_kernel_size, min_area_boundingbox = self.class_color_info_map[label]
            color = np.array(color)

            # Create a mask for the current color
            mask = cv2.inRange(img_rgb, color, color)

            # To reduce the incontinuities in the mask we will perfomr a dilation
            kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

            # Find contours for the masked region
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Filter the contours and add them to a list+ we also correct for the dilation that we did previously
            boundingboxes = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w*h > min_area_boundingbox:
                    boundingboxes.append([x + int(dilation_kernel_size/2), y + int(dilation_kernel_size/2), w - dilation_kernel_size, h - dilation_kernel_size])

            boundingboxes_per_class[label] = boundingboxes

        return boundingboxes_per_class

    def convert_data(self, first_index, train_percentage):
        # We print some information
        print('Converting data in folder ' + self.input_dir)

        # We keep track of this index to make sure that we know what the last index of the image was that we added.
        current_index = first_index

        # We create our folders and our .yaml file
        self.create_data_folders()
        self.create_yaml_file()

        # We define our segmented images and normal images foldes
        seg_folder = os.path.join(self.input_dir, 'segmentation image')
        normal_folder = os.path.join(self.input_dir, 'custom_data')

        # We get all the image files
        seg_image_files = [f for f in os.listdir(seg_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
        normal_image_files = [f for f in os.listdir(normal_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # we check wether the amount of images is the same
        if len(seg_image_files) != len(normal_image_files):
            print('Error in folder ' + self.input_dir + ': Amount of segmented and normal images should be the same!\n')
        else:
            # We define how many training images we have
            num_training_images = int(len(seg_image_files)*train_percentage)
    
            # We assume that the images have the same index at the end
            for image_index in range(len(seg_image_files)):
                # We define wether we want this image to be a train image or a validation image
                mode = 'train' if image_index < num_training_images else 'val'
    
                # We identify all the paths that we need
                seg_image_path = os.path.join(seg_folder, seg_image_files[image_index])
                normal_image_input_path = os.path.join(normal_folder, normal_image_files[image_index])
                normal_image_output_path = os.path.join('datasets', self.output_dir, 'images', mode, str(current_index) + '.png')
                labels_ouput_path = os.path.join('datasets', self.output_dir, 'labels', mode, str(current_index) + '.txt')
    
                # We read the image get the width and the height and write the normal image to the correct path
                normal_image = cv2.imread(normal_image_input_path)
                height, width, _ = normal_image.shape
                cv2.imwrite(normal_image_output_path, normal_image)
    
                # We get the bounding boxes from the image
                bounding_boxes = self.get_2D_bounding_box_from_segmeted_image(seg_image_path)
    
                # We write the bounding box data in the correct format to the .txt file
                with open(labels_ouput_path, 'w') as file:
                    class_index = 0
                    for _ , bboxes in bounding_boxes.items():
                        if len(bboxes) != 0:
                            for bbox in bboxes:
                                # We normalize the bbox values
                                x, y, w, h = bbox
                                norm_x = x / width
                                norm_y = y / height
                                norm_w = w / width
                                norm_h = h / height
    
                                # We define the string and write it to the .txt file
                                bbox_string = str(class_index) + ' ' + str(norm_x) + ' ' + str(norm_y) + ' ' + str(norm_w) + ' ' + str(norm_h)
                                file.write(bbox_string + '\n')
                        class_index += 1
                current_index += 1
            print('Succesfully converted the data!\n')
        return current_index


    def draw_rectangles(self, image_path, bounding_boxes):
        image = cv2.imread(image_path)

        for label in bounding_boxes.keys():
            if len(bounding_boxes[label]) != 0:
                for boundingbox in bounding_boxes[label]:
                    x, y, width, height = boundingbox
                    cv2.rectangle(image, (x, y), (x + width, y + height), (0,0,225), 2)

        # Display the image with rectangles
        cv2.imshow("Image with Rectangles", image)
        cv2.waitKey(0)  # Wait for a key press to close the window
        cv2.destroyAllWindows()

        # Optionally save the modified image
        cv2.imwrite('output_image.jpg', image)

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
            'nc': len(self.class_color_info_map.keys()),        # Number of classes
            'names': list(self.class_color_info_map.keys())     # List of class names
        }

        # We write the data to the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)
        print(f"YAML file created: {yaml_path}")