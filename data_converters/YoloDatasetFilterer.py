import os
import shutil
import pygame
from data_converters import DataConverter

class YoloDatasetFilterer(DataConverter.DataConverter):
    def __init__(self, input_dir, output_dir, classes):
        super().__init__(input_dir, output_dir, classes)
        
        self.create_yaml_file()
        self.create_data_folders()

    def draw_ractangles(self, screen, bounding_boxes, highligted_rectangle, width, height):
        for bbox_index in range(len(bounding_boxes)):
            class_number, dimentions = bounding_boxes[bbox_index]
            x_center, y_center, w, h = dimentions

            x = int((x_center - (w/2)) * width)
            y = int((y_center - (h/2)) * height)
            w = int(w * width)
            h = int(h * height)
            label = self.classes[int(class_number)]

            if bbox_index == highligted_rectangle:
                pygame.draw.rect(screen, (0, 255, 0), (x, y, w, h), 3)
            else:
                pygame.draw.rect(screen, (255, 0, 0), (x, y, w, h), 3)

    def filter_data(self, start_index, estimated_image_size):
        # We initialize pygame that we will use to show the images
        pygame.init()
        width, height = estimated_image_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Image filterer')

        filtered_image_index = {'train': 0, 'val': 0}
        for mode in ['train', 'val']:
            # We define the data folders
            images_input_path = os.path.join('datasets', self.input_dir, 'images', mode)
            labels_input_path = os.path.join('datasets', self.input_dir, 'labels', mode)
            images_output_path = os.path.join('datasets', self.output_dir, 'images', mode)
            labels_output_path = os.path.join('datasets', self.output_dir, 'labels', mode)

            # We get the file names and pair them correspondingly of data samples
            # We assume that the image and corresponding file have the same name and will be ordered in the same way
            image_files = [f for f in os.listdir(images_input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            label_files = [f for f in os.listdir(labels_input_path) if f.endswith(('.txt'))]
            file_list = [(image_files[file_index] ,label_files[file_index]) for file_index in range(len(image_files))]

            # We loop over all the files and order them
            files_index = 0
            while files_index < len(file_list):
                image_file, label_file = file_list[files_index]
                # We define the specific data folders
                image_input_path = os.path.join(images_input_path, image_file)
                label_input_path = os.path.join(labels_input_path, label_file)
                image_output_path = os.path.join(images_output_path, str(filtered_image_index[mode]) + '.png')
                label_output_path = os.path.join(labels_output_path, str(filtered_image_index[mode]) + '.txt')

                # We load and draw the image on the screen
                image = pygame.image.load(image_input_path) 
                image = pygame.transform.scale(image, (width, height))
                screen.blit(image, image.get_rect(center = screen.get_rect().center))

                # Get the bounding boxes in the file
                bounding_boxes = self.get_bounding_boxes(label_input_path)

                # We loop over each rectangle to be able to remove or keep it
                i = 0
                loop = True
                make_file = False
                while loop and  i < len(bounding_boxes):
                    # We draw the rectangles and update the screen
                    screen.blit(image, image.get_rect(center = screen.get_rect().center))
                    self.draw_ractangles(screen, bounding_boxes, i, width, height)
                    pygame.display.flip()

                    # Based on the key that is pressed we add or remove the boundingbox
                    key_number = self.get_key_press()
                    if key_number == 1:
                        loop = False
                        make_file = True
                    elif key_number == 2:
                        loop = False
                        make_file = False
                    elif key_number == 3:
                        bounding_boxes.remove(bounding_boxes[i])
                        make_file = True
                    elif key_number == 4:
                        make_file = True
                        i += 1
                    elif key_number == 5:
                        files_index = max(files_index - 2, 0)
                        filtered_image_index[mode] = max(filtered_image_index[mode] - 1, 0)
                        loop = False
                        make_file = False
                
                # If make_file is true we copy the image and craete the correct label file in the output directory
                if make_file:
                    shutil.copy(image_input_path, image_output_path)
                    with open(label_output_path, "w") as file:
                        for class_number, coordinates in bounding_boxes:
                            x_center, y_center, w, h = coordinates

                            bbox_string = str(class_number) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(w) + ' ' + str(h) + '\n'
                            file.write(bbox_string)
                
                    print('Moved '+ image_input_path + ' to ' + image_output_path)
                    print('Created label file at ' + label_output_path)
                    filtered_image_index[mode] += 1
                files_index += 1

        # Quit pygame
        pygame.quit()
    
    def get_bounding_boxes(self, label_input_path):
        # We get the boundingboxes from the file
        bounding_boxes = list()
        with open(label_input_path, 'r') as file:
            for line in file:
                class_number, x_center, y_center, w, h = [float(i) for i in line.split()]
                label = self.classes[int(class_number)]

                bounding_boxes.append((class_number, [x_center, y_center, w, h]))
        
        return bounding_boxes
    

    def get_key_press(self):
        # We keep waiting for an enter or backspace key to be pressed
        scanning =True
        while scanning:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    scanning = False
                elif event.type == pygame.KEYDOWN:
                     # We check what key is pressed and return a value based on it
                    if event.key == pygame.K_RETURN:
                        scanning = False
                        return 1
                    if event.key == pygame.K_BACKSPACE:
                        scanning = False
                        return 2
                    if event.key == pygame.K_DOWN:
                        scanning = False
                        return 3
                    if event.key == pygame.K_UP:
                        scanning = False
                        return 4
                    if event.key == pygame.K_LEFT:
                        scanning = False
                        return 5
                