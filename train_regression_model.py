from models import RegressionModel
import BoundingBoxRegressionDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser(description="training argument parsers ")
parser.add_argument("dataset", type=str, help="the location of the dataset")
parser.add_argument("number_of_epochs", type=int, help="number of epochs")
parser.add_argument("weights_location", type=str, help='the location where the weights of the run will be saved')

args = parser.parse_args()

number_of_epochs = args.number_of_epochs
dataset = args.dataset
weights_location = args.weights_location

model = RegressionModel.RegressionModel()

train_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset(dataset, train=True)
val_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset(dataset, train=False)

train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

print(train_dataset.__getitem__(1))

model.train_model(optimizer, criterion, train_loader, val_loader, number_of_epochs, weights_location)