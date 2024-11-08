import os
import shutil
import pygame
from data_converters import DataConverter

class YoloDatasetFilterer(DataConverter.DataConverter):
    def __init__(self, input_dir, output_dir, classes):
        super().__init__(input_dir, output_dir, classes)
        
        self.create_yaml_file()
        self.create_data_folders()

    def filter_data(self, start_index, estimated_image_size):
        # We initialize pygame that we will use to show the images
        pygame.init()
        width, height = estimated_image_size
        screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Image filterer')

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

                # We load and draw the image on the screen
                image = pygame.image.load(image_input_path) 
                image = pygame.transform.scale(image, (width, height))
                screen.blit(image, image.get_rect(center = screen.get_rect().center))

                # We draw the boundingboxes on the screen
                with open(label_input_path, 'r') as file:
                    for line in file:
                        class_number , x, y, w, h = [float(i) for i in line.split()]
                        x = int((x - (w/2)) * width)
                        y = int((y - (h/2)) * height)
                        w = int(w * width)
                        h = int(h * height)
                        label = self.classes[int(class_number)]

                        # We draw the bounding box
                        pygame.draw.rect(screen, (255, 0, 0), (x, y, w, h), 3)

                        # Add the class name
                        #text_position = (x1, y1 - 10)  # Position the text above the box
                        #cv2.putText(img, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # We update the display
                pygame.display.flip()

                # We keep waiting for an enter or backspace key to be pressed
                scanning =True
                while scanning:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            scanning = False
                        elif event.type == pygame.KEYDOWN:
                             # We check if the key pressed matches any key in our directories dictionary
                            if event.key == pygame.K_RETURN:
                                # If enter is pressed we save the images otherwise (if backspace is pressed) we do not
                                scanning = False
                                shutil.copy(image_input_path, image_output_path)
                                shutil.copy(label_input_path, label_output_path)
                                print('Moved '+ image_input_path + ' to ' + image_output_path)
                                print('Moved '+ label_input_path + ' to ' + label_output_path)
                            if event.key == pygame.K_BACKSPACE:
                                # If backspace is pressed we move to the next image
                                scanning = False
        # Quit pygame
        pygame.quit()