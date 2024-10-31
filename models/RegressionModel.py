import torch
import torch.nn as nn

# We will use this model to map the 2D bouding boxes to a 3D one.
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # The input layer of the model has 5 features (2D bbox (xy, xy) (4), class number (1))
        self.lin1 = nn.Linear(5, 256)
        self.lin2 = nn.Linear(256, 256)
        # The output of the model has 8 features (3D location (x, y, z) (3), 3D size (w, h, l), rotation_angle)
        self.lin3 = nn.Linear(256, 8)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
    def train_model(self, optimizer, criterion, train_loader, validation_loader, epochs):
        for epoch in range(epochs):
            self.train()
            for inputs, targets in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs, targets)  

                # We compute the gradients and update the weights
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')
            self.validate_model(validation_loader, criterion)

    def validate_model(self, validation_loader, criterion):
        self.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():  # Disable gradient calculation
            for inputs, targets in validation_loader:
                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs, targets)
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        print(f'Validation Loss: {avg_loss:.4f}')