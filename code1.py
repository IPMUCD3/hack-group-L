import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
# Load data
data = np.load('avg_coefs_z1.npy')  # scattering coefficients. shape: (num_simulations, num_patches, num_coefficients)
data = np.mean(data, axis=1)
data = data[1:]
targets = np.load('v1_s8_ordered.npy')[1:, 2][:, None]  #S8. shape: (num_simulations, 1)
# print(targets)
print(data.shape, targets.shape)
# data = torch.linspace(0, 1, 100)[:, None]
# targets = torch.linspace(0, 1, 100)[:, None]
# Convert to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
targets = torch.tensor(targets, dtype=torch.float32)
# shuffle the data and targets along the first dimension, to mix the sims up
indices = torch.randperm(data.shape[0])
data = data[indices]
targets = targets[indices]
print(data.shape, targets.shape)
# Split into training and validation sets, so we have a section of different S8 values
num_train = int(0.8 * len(data))
train_data, val_data = data[:num_train], data[num_train:]
train_targets, val_targets = targets[:num_train], targets[num_train:]
print(train_data.shape, val_data.shape)
print(train_targets.shape, val_targets.shape)
num_coeffs = data.shape[-1]
num_params = targets.shape[-1]
# Create DataLoader instances
batch_size = 32
train_dataset = TensorDataset(train_data, train_targets)
val_dataset = TensorDataset(val_data, val_targets)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
print(len(train_dataset))
# Neural Network Definition
class NeuralNet(nn.Module):
    def __init__(self, num_features, num_targets):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(num_features, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_targets)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
# Model, Loss and Optimizer
model = NeuralNet(num_coeffs, num_params)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# Training Loop
def train_model(num_epochs):
    train_losses, val_losses = [], []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_losses.append(val_loss / len(val_loader))
        if epoch % 100 == 0:
          print(
              f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')
    return train_losses, val_losses
# Train the model
num_epochs = 1000
train_losses, val_losses = train_model(num_epochs)
# Plotting the training and validation loss
plt.plot(range(num_epochs), train_losses, label='Train')
plt.plot(range(num_epochs), val_losses, label='Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save the model
torch.save(model.state_dict(), 'model.pth')
