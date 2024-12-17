from models import RegressionModel
import torch
import BoundingBoxRegressionDataset
import argparse

parser = argparse.ArgumentParser(description="training argument parsers ")
parser.add_argument("model_path", type=str, help="the path of model where the saved weights are stored")
parser.add_argument("visualization_amount", type=int, help="the amount of visualized images")

args = parser.parse_args()

model_path = args.model_path
visualization_amount = args.visualization_amount

model = RegressionModel.RegressionModel()

model_weigths_path = model_path + '/best.pth'
state_dict = torch.load(model_weigths_path)
model.load_state_dict(state_dict)

val_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset('datasets/kitti_transformed', train=False)

model.visualize_model(visualization_amount, val_dataset)