from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

class ObjectLocalizationModel:

    def __init__(self):
        self.model = YOLO('yolov8s.pt')

    def forward(self, image_path):
        results = self.model(image_path)
        return results

    def visualize_results(self, results, image_path):
        object_classes = results[0].boxes.cls.to('cpu').tolist()
        bboxes_xyxy = results[0].boxes.xyxy.to('cpu').tolist()
        all_class_names = results[0].names
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

        # We plot the image
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

    def train_model(self, yaml_file, num_epochs, batch_size = 16):
        results = self.model.train(data=yaml_file, epochs= num_epochs, imgsz=640, batch= batch_size)
        return results