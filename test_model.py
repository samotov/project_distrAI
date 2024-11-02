from models import ObjectLocalizationModel
# import argparse

# parser = argparse.ArgumentParser(description="Argument parsers ")
# parser.add_argument("dataset", type=str, help="the location of the dataset")

# args = parser.parse_args()

# dataset = args.dataset

dataset = 'datasets/captured_data/cloudy_noon/custom_data'

yolomodel = ObjectLocalizationModel.ObjectLocalizationModel('best.pt')

yolomodel.testYoloModel(dataset)