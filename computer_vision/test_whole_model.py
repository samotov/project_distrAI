from models import RegressionModel
from models import ObjectLocalizationModel
import torch
import BoundingBoxRegressionDataset
import argparse

parser = argparse.ArgumentParser(description="Argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the images dataset")
parser.add_argument("object_localization_model_path", type=str, help="the path of the object localization model where the saved weights are stored")
parser.add_argument("regression_model_path", type=str, help="the path of regression model where the saved weights are stored")
parser.add_argument("visualization_amount", type=int, help="the amount of visualized images")

args = parser.parse_args()

dataset = args.dataset
object_localization_model_path = args.object_localization_model_path
regression_model_path = args.regression_model_path
visualization_amount = args.visualization_amount

yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')

yolomodel.testYoloModel(dataset)




model = RegressionModel.RegressionModel()

model_weigths_path = model_path + '/best.pth'
state_dict = torch.load(model_weigths_path)
model.load_state_dict(state_dict)

val_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset('datasets/kitti_transformed', train=False)

model.visualize_model(visualization_amount, val_dataset)