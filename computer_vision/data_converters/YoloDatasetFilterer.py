import os
import shutil
import pygame
import time
import yaml
from PIL import Image
from data_converters import DataConverter

class YoloDatasetFilterer(DataConverter.DataConverter):
    def __init__(self, source_yaml_file, output_dir):
        with open(source_yaml_file, 'r') as file:
            dataset_input_config = yaml.safe_load(file)
        
        super().__init__(output_dir, dataset_input_config['names'])
        
        self.input_dir = dataset_input_config['train'].partition('\\')[0]
        self.create_yaml_file()
        self.create_data_folders()

    def draw_ractangles(self, screen, bounding_boxes, highligted_rectangle, width, height):
        for bbox_index in range(len(bounding_boxes)):
            class_number, dimentions = bounding_boxes[bbox_index]
            x_center, y_center, w, h = dimentions

            # We get the boundingbox data
            x = int((x_center - (w/2)) * width)
            y = int((y_center - (h/2)) * height)
            w = int(w * width)
            h = int(h * height)
            label = self.classes[int(class_number)]

            # Based on the boolean highlighted_rectangle we use a different color
            color = (0, 255, 0) if bbox_index == highligted_rectangle else (255, 0, 0)

            # We draw the boundingbox and its class above it based on the color
            pygame.draw.rect(screen, color, (x, y, w, h), 3)
            font = pygame.font.Font(None, 20)  # None for default font, 36 is the font size
            text_surface = font.render(str(label), True, color)  # True for anti-aliasing
            screen.blit(text_surface, (x, y - 15))

    def filter_data(self, start_index_org_data, filter_image_index_train, filter_image_index_val):
        # We initialize the indexes for the next validation and training image
        filtered_image_index = {'train': filter_image_index_train, 'val': filter_image_index_val}

        # We get a big list of all the files
        file_list = []
        for mode in ['train', 'val']:
            # We define the data folders
            images_input_path = os.path.join('datasets', self.input_dir, 'images', mode)
            labels_input_path = os.path.join('datasets', self.input_dir, 'labels', mode)

            # We get the file names and pair them with the corresponding data samples and add wether its a validation or training image
            # We assume that the image and corresponding file have the same name and will be ordered in the same way
            image_files = [f for f in os.listdir(images_input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            label_files = [f for f in os.listdir(labels_input_path) if f.endswith(('.txt'))]
            file_list += [(image_files[file_index] ,label_files[file_index], mode) for file_index in range(len(image_files))]

        # We get the image size of the first image and assume that they are all the same
        image_file = os.path.join('datasets', self.input_dir, 'images', file_list[0][2], file_list[0][0])
        with Image.open(image_file) as img:
            # Get image size
            width, height = img.size

        # We initialize pygame that we will use to show the images
        pygame.init()
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Image filterer')

        # We loop over all the files and order them
        files_index = start_index_org_data
        while files_index < len(file_list):
            image_file, label_file, mode = file_list[files_index]
            # We define the specific data folders
            image_input_path = os.path.join('datasets', self.input_dir, 'images', mode, image_file)
            label_input_path = os.path.join('datasets', self.input_dir, 'labels', mode, label_file)
            image_output_path = os.path.join('datasets', self.output_dir, 'images', mode, str(filtered_image_index[mode]) + '.png')
            label_output_path = os.path.join('datasets', self.output_dir, 'labels', mode, str(filtered_image_index[mode]) + '.txt')

            # We load and draw the image on the screen
            image = pygame.image.load(image_input_path) 
            image = pygame.transform.scale(image, (width, height))
            screen.blit(image, image.get_rect(center = screen.get_rect().center))

            # Get the bounding boxes in the file
            bounding_boxes = self.get_bounding_boxes(label_input_path)

            # We define the file_index_difference to be 1 (this can be changed to move forward or backwards)
            file_index_difference = 1

            # We loop over each rectangle to be able to remove or keep it
            i = 0
            loop = True
            make_file = False

            # Depending on the amount of bounding boxes we perform a different while loop
            if len(bounding_boxes) != 0:
                while loop and  i < len(bounding_boxes):
                    # We draw the rectangles and update the screen
                    screen.blit(image, image.get_rect(center = screen.get_rect().center))
                    self.draw_ractangles(screen, bounding_boxes, i, width, height)
                    pygame.display.flip()

                    # Based on the key that is pressed we add or remove the boundingbox
                    key_number = self.get_key_press()
                    time.sleep(0.1)
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
                        # We go to the previous file
                        file_index_difference = -1
                        filtered_image_index[mode] = max(filtered_image_index[mode] - 1, 0)
                        loop = False
                        make_file = False
                    elif key_number == 6:
                        # We change the boundingbox label to traffic light green
                        bounding_boxes[i] = (self.classes.index("traffic light green"), bounding_boxes[i][1])
                        i += 1
                        loop = True
                        make_file = True
                    elif key_number == 7:
                        # We change the boundingbox label to traffic light orange
                        bounding_boxes[i] = (self.classes.index("traffic light orange"), bounding_boxes[i][1])
                        i += 1
                        loop = True
                        make_file = True
                    elif key_number == 8:
                        # We change the boundingbox label to traffic light red
                        bounding_boxes[i] = (self.classes.index("traffic light red"), bounding_boxes[i][1])
                        i += 1
                        loop = True
                        make_file = True
                    elif key_number == 9:
                        # We change the boundingbox label to traffic light unimportant
                        bounding_boxes[i] = (self.classes.index("traffic light unimportant"), bounding_boxes[i][1])
                        i += 1
                        loop = True
                        make_file = True
            else:
                while loop:
                    # We update the screen
                    pygame.display.flip()

                    # Based on the key that is pressed we add or remove the boundingbox
                    key_number = self.get_key_press()

                    if key_number == 1:
                        loop = False
                        make_file = True
                    elif key_number == 2:
                        loop = False
                        make_file = False
                    elif key_number == 5:
                        # We go to the previous file
                        file_index_difference = -1
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
                
                print('Copied '+ image_input_path + ' to ' + image_output_path)
                print('Created label file at ' + label_output_path)
                filtered_image_index[mode] += 1
            # We update the files index
            files_index += file_index_difference

            # We print some debug information
            print("File index: [" + str(files_index) + "/" + str(len(file_list)) + "] Use this this index in the config to start from this index input image")
            print("Next training image index: ", filtered_image_index['train'], " Use this in the config to make sure you don't write over existing data!")
            print("Next validation image index: ", filtered_image_index['val'], "Use this in the config to make sure you don't write over existing data!")
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
                    if event.key == pygame.K_g:
                        scanning = False
                        return 6
                    if event.key == pygame.K_o:
                        scanning = False
                        return 7
                    if event.key == pygame.K_r:
                        scanning = False
                        return 8
                    if event.key == pygame.K_u:
                        scanning = False
                        return 9