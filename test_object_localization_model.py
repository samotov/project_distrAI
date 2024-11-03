from models import ObjectLocalizationModel
import argparse

parser = argparse.ArgumentParser(description="Argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the dataset")

args = parser.parse_args()

dataset = args.dataset

yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('models/object_localization_weights/best.pt')

yolomodel.testYoloModel(dataset)