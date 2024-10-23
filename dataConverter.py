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

    def convert_data(self, first_index):
        # We keep track of this index to make sure that we know what the last index of the image was that we added.
        current_index = first_index

        # We create our folders and our .yaml file
        self.create_data_folders()
        self.create_yaml_file()

        # We get all the image files
        image_files = [f for f in os.listdir(self.input_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

        for image in image_files:
            image_path = os.path.join(self.input_dir, "segmentation image",image)
            bounding_boxes = self.get_2D_bounding_box_from_segmeted_image(image_path)

            for label in bounding_boxes.keys():
                print('test')
                # add bounding box data and image data to the correct folder
            current_index += 1
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
            folder_path = os.path.join(self.output_dir, folder)

            # Check if the folder already exists, if not, create it
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Created folder: {folder_path}")
            else:
                print(f"Folder already exists: {folder_path}")

    def create_yaml_file(self):
        yaml_path = os.path.join(self.output_dir, "custom_data.yaml")
        # We define the content for the YAML file
        data = {
            'train': os.path.join(self.output_dir, "train"),    # Path to the training data
            'val': os.path.join(self.output_dir, "val"),        # Path to the validation data
            'nc': len(self.class_color_info_map.keys()),        # Number of classes
            'names': self.class_color_info_map.keys()           # List of class names
        }

        # We write the data to the YAML file
        with open(yaml_path, 'w') as file:
            yaml.dump(data, file)
        print(f"YAML file created: {yaml_path}")
