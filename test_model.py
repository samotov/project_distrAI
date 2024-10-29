import ObjectLocalizationModel
import argparse

parser = argparse.ArgumentParser(description="Argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the dataset")

args = parser.parse_args()

dataset = args.dataset

yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('best.pt')
yolomodel.testYoloModel(dataset)