import RegressionModel
import ObjectLocalizationModel
import torch.nn as nn
import torch.optim as optim
import os

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
    testYoloModel()

    model = RegressionModel.RegressionModel()

    # These are great choices when training a regression model
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), LEARNING_RATE)

    #model.train_model(optimizer, criterion)

if __name__ == "__main__":
    main()