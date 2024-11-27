from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import os

class ObjectLocalizationModel:

    def __init__(self, weights):
        if weights == None:
            self.model = YOLO('yolov8s.pt')
        else:
            self.model = YOLO(weights)

    def forward(self, image_path):
        results = self.model(image_path)
        return results

    def train_model(self, yaml_file, num_epochs, batch_size = 16, image_conversion_size = 640):
        results = self.model.train(data=yaml_file, epochs= num_epochs, imgsz=image_conversion_size, batch= batch_size)
        return results


    def testYoloModel(self, test_folder):
        # get the image files
        image_files = [f for f in os.listdir(test_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

        # Loop over the images
        for image_file in image_files:
            image_path = os.path.join(test_folder, image_file)
            results = self.forward(image_path)
            self.visualize_results(results, image_path)
    

    def visualize_results(self, results, image_path):
        object_classes = results[0].boxes.cls.to('cpu').tolist()
        bboxes_xyxy = results[0].boxes.xyxy.to('cpu').tolist()
        all_class_names = results[0].names
        print(all_class_names)
        class_names = [all_class_names[int(x)] for x in object_classes]
        img = cv2.imread(image_path)

        for i in range(len(object_classes)):
            x_min, y_min, x_max, y_max = [int(x) for x in bboxes_xyxy[i]]
            object_class = class_names[i]

            # Add a rectangle
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # Add the class name
            text_position = (x_min, y_min - 10)  # Position the text above the box
            cv2.putText(img, object_class, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # We plot the image and wait 1 sec before we plot the next one
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show(block = False)
        plt.pause(1)
        plt.close()
    
    def visualize_data(self, amount, data_path):
        for i in range(amount):
            # We initialize the image path and results parg
            image_path = os.path.join('datasets', data_path, 'images', 'train', str(i) + '.png')
            results_path = os.path.join('datasets', data_path, 'labels', 'train', str(i) + '.txt')

            # We get the lines of the results file
            results_file = open(results_path, 'r')
            results_lines = results_file.readlines()

            # We read the image
            img = cv2.imread(image_path)
            height, width, _ = img.shape

            # We read each line and plot the boundingbox on the image
            for result_line in results_lines:
                # We extract the result and draw the boundingbox
                _ , x_center, y_center, w, h = [float(result_value) for result_value in result_line.split()]

                x_min = int((x_center - (w/2)) * width)
                y_min = int((y_center - (h/2)) * height)
                x_max = int((x_center + (w/2)) * width)
                y_max = int((y_center + (h/2)) * height)

                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

            # we close the files
            results_file.close()

            # We plot the image and wait 1 sec before we plot the next one
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.show(block = False)
            plt.pause(1)
            plt.close()
