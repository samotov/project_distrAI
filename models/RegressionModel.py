import torch
import torch.nn as nn
import wandb
from Boundingbox3D import BoundingBox3D
import matplotlib.pyplot as plt

# We will use this model to map the 2D bouding boxes to a 3D one.
class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        # The input layer of the model has 5 features (2D bbox (xy, xy) (4), class number (1))
        self.lin1 = nn.Linear(5, 256)
        self.lin2 = nn.Linear(256, 256)
        # The output of the model has 2 features (3D location point1 (x, y, z) (3), 3D location point2 (w, h, l) (3)))
        self.lin3 = nn.Linear(256, 6)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        x = self.lin3(x)
        return x
    
    def train_model(self, optimizer, criterion, train_loader, validation_loader, epochs, model_save_path):
        # Initialize Weights & Biases run
        wandb.init(project="regression_model")
        wandb.watch(self, criterion, log="all", log_freq=10)  # Track gradients and model parameters

        best_loss = float('inf')  # Variable to track the best validation loss
        
        for epoch in range(epochs):
            self.train()  # Set model to training mode
            running_loss = 0.0

            for inputs, targets in train_loader:
                # Zero the gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = self.forward(inputs)

                # Compute loss
                loss = criterion(outputs, targets)  

                # Backpropagation and weight update
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            # Average loss for this epoch
            avg_train_loss = running_loss / len(train_loader)

            # Validation step
            avg_val_loss = self.validate_model(validation_loader, criterion)

            # Log training and validation losses to W&B
            wandb.log({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

            # Print progress
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

            # Save the last weights
            torch.save(self.state_dict(), f"{model_save_path}/last.pth")

            # Save best model weights if validation loss improves
            if avg_val_loss < best_loss:
                best_loss = avg_val_loss
                torch.save(self.state_dict(), f"{model_save_path}/best.pth")
                print(f"New best model saved with validation loss: {best_loss:.4f}")

        # End W&B run
        wandb.finish()

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
        return avg_loss


    def visualize_model(self, amount, dataset):
        amount = min(len(dataset), amount)
        for i in range(amount):
            features, ground_truth = dataset.__getitem__(i)
            output = self.forward(features)
            # We get the values of the output and the ground truth
            x1_truth, y1_truth, z1_truth, x2_truth, y2_truth, z2_truth = ground_truth.tolist()
            x1_out, y1_out, z1_out, x2_out, y2_out, z2_out = output.tolist()

            # We compose 2 bounding boxes
            ground_truth_bbox = BoundingBox3D((x1_truth, y1_truth, z1_truth), (x2_truth, y2_truth, z2_truth))
            output_bbox = BoundingBox3D((x1_out, y1_out, z1_out), (x2_out, y2_out, z2_out))

            # We plot the bounding boxes
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ground_truth_bbox.visualize_boundingbox(ax, 'b')
            output_bbox.visualize_boundingbox(ax, 'r')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([-20, 20])
            ax.set_ylim([-20, 20])
            ax.set_zlim([0, 40])
            plt.show()




