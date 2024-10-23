import RegressionModel
import ObjectLocalizationModel
import DataConverter
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

LEARNING_RATE = 0.001

def testYoloModel():
    yolo_model = ObjectLocalizationModel.ObjectLocalizationModel()
    image_folder = 'captured_data/cloudy_night/custom_data'

    # get the image files
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Loop over the images
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        results = yolo_model.forward(image_path)
        yolo_model.visualize_results(results, image_path)

def main():
    # This map contains all the info as: label: [color, dilation_kernel_size, min_area_boundingbox]
    class_color_info_map = {'car': [np.array([0, 0, 142]), 10, 250],
                'motorcycle': [np.array([0, 0, 230]), 10, 250],
                'truck': [np.array([0, 0, 70]), 10, 250],
                'pedestrian': [np.array([220, 20, 60]), 8, 200],
                'traffic signs': [np.array([220, 220, 0]), 1, 150],
                'traffic lights': [np.array([250, 170, 30]), 1, 150]}
    
    data_converter = DataConverter.DataConverter("captured_data/noon", "test", class_color_info_map)

    data_converter.convert_data(1, 0.8)
    
    

if __name__ == "__main__":
    main()