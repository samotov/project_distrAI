from models import ObjectLocalizationModel
import argparse

# Read the arguments passed from the command
parser = argparse.ArgumentParser(description="Argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the dataset")
args = parser.parse_args()
dataset = args.dataset

# Test the yolomodel
yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')
yolomodel.testYoloModel(dataset)