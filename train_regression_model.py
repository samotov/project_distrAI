from models import RegressionModel
import BoundingBoxRegressionDataset

from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

model = RegressionModel.RegressionModel()

train_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset('datasets/kitti_transformed', train=True)
val_dataset = BoundingBoxRegressionDataset.BoundingBoxRegressionDataset('datasets/kitti_transformed', train=False)

train_dataset.visualize_data(5)

"""
train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = True)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train_model(optimizer, criterion, train_loader, val_loader, 100)

"""
