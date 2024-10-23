import torch.nn as nn
import torch.optim as optim

# We will use this model to map the 2D bouding boxes to a 3D one.
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # The input layer of the model has  features (2D bbox (xy, xy) (4), class 1-hot-encoding)
        self.lin1 = nn.Linear(20,100)
        self.lin2 = nn.Linear(100,16)

    def forward(self, x):
        x = nn.ReLU(self.lin1(x))
        x = nn.ReLU(self.lin2(x))
        return x
    
    def train_model(self, optimizer, criterion, train_loader, epochs):
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs)  

                # We compute the gradients and update the weights
                loss.backward()
                optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    def validate_model(self, validation_loader):
        print('test')