from models import ObjectLocalizationModel
import argparse

def train_model(number_of_epochs, dataset):
    yolomodel = ObjectLocalizationModel.ObjectLocalizationModel()

    yolomodel.train_model(dataset + '.yaml', number_of_epochs, 16)

parser = argparse.ArgumentParser(description="number of epochs argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the dataset")
parser.add_argument("number_of_epochs", type=int, help="number of epochs")

args = parser.parse_args()

number_of_epochs = args.number_of_epochs
dataset = args.dataset
    
train_model(number_of_epochs, dataset)