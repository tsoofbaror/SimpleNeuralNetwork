import pandas as pd

from data_preprocess import preprocess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from plots import plot_results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# הגדרת אובייקט - רשת נוירונים
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        # עניינים טכניים
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        # הגדרת שכבת כניסה
        self.layers.append(nn.Linear(input_size, hidden_size[0]))  # שכבת הכניסה
        self.layers.append(nn.ReLU())  # אקטיבציה אחרי שכבת הכניסה

        # שכבות חבויות
        for i in range(len(hidden_size) - 1):
            self.layers.append(nn.Linear(hidden_size[i], hidden_size[i + 1])) # שכבה
            self.layers.append(nn.ReLU()) # אקטיבציה

        # שכבת יציאה
        self.layers.append(nn.Linear(hidden_size[-1], num_classes))

    def forward(self, x):
        # מעבר קלט בכל השכבות
        for layer in self.layers:
            x = layer(x)
        return x


train_x, train_y, val_x, val_y, test_x, test_y, mapping_dict = preprocess("train.csv", "test.csv")
print(mapping_dict)
exit()
train_x = train_x.to(device)
train_y = train_y.to(device)
val_x = val_x.to(device)
val_y = val_y.to(device)

# Set the hyperparameters
input_size = train_x.shape[1]
hidden_size = [256, 1024, 1024, 512, 32]
num_classes = 1
learning_rate = 0.0001
num_epochs = 100
# Set your batch size
batch_size = 516

# Compute the number of batches
n_batches = int(np.ceil(len(train_x) / batch_size))

# Create the neural network object
model = NeuralNetwork(input_size, hidden_size, num_classes).to(device)

# Define the loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define the lists to store the results
train_losses = []
val_losses = []
val_accs = []

# Define the lists to store the results
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(num_epochs):

    # Shuffle your data at the start of each epoch
    indices = np.arange(len(train_x))
    np.random.shuffle(indices)
    train_x = train_x[indices]
    train_y = train_y[indices]

    for i in range(n_batches):
        # Compute the start and end indices for the current batch
        start = i * batch_size
        end = min(start + batch_size, len(train_x))

        # Extract the current batch
        batch_x = train_x[start:end]
        batch_y = train_y[start:end]

        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs.squeeze(1), batch_y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Evaluate on validation set after each epoch
    model.eval()
    with torch.no_grad():
        # Compute training accuracy
        outputs = model(train_x).squeeze(1)
        train_preds = (torch.sigmoid(outputs) > 0.5).float()
        train_correct = (train_preds == train_y).sum().item()
        train_acc = train_correct / len(train_y)

        val_outputs = model(val_x).squeeze(1)
        val_loss = criterion(val_outputs, val_y)

        # Convert outputs to predicted class
        val_preds = (val_outputs > 0.5).float()  # Changed this line
        val_correct = (val_preds == val_y).sum().item()
        val_acc = val_correct / len(val_y)

    # Switch back to training mode
    model.train()

    # Store losses and accuracy
    train_accs.append(train_acc)
    train_losses.append(loss.item())
    val_losses.append(val_loss.item())
    val_accs.append(val_acc)

    # Print the validation loss and accuracy after each epoch
    print(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc:.4f}")

# Plot the training losses, validation losses, and validation accuracies
plot_results(train_losses, val_losses, val_accs, train_accs)
