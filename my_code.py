#step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Load data
data = pd.read_csv('data.csv')

# Select features and target
features = data[['step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']]
target = data['isFraud']  # Changed to a Series instead of DataFrame

# Encode categorical data safely
features = features.copy()
features['type'] = LabelEncoder().fit_transform(features['type'])

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)  # Use the same scaler for test data

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Reshape labels to meet BCELoss requirements
y_train = y_train.unsqueeze(1)
y_test = y_test.unsqueeze(1)

# Create DataLoader instances
train_data = TensorDataset(X_train, y_train)
test_data = TensorDataset(X_test, y_test)

batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Define the Neural Network Model
class FraudNet(nn.Module):
    def __init__(self):
        super(FraudNet, self).__init__()
        self.fc1 = nn.Linear(7, 16)  # 7 input features, for the different titles in the dataset
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)  # Output layer, converting it back to 1 answer

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))  # Using sigmoid for the final output
        return x

# Training the Model
model = FraudNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of epochs
epochs = 5

# Training loop
for epoch in range(epochs):
	model.train()  # Set the model to training mode
	total_loss = 0
	for inputs, labels in train_loader:
		optimizer.zero_grad()  # Reset gradients, so previous gradients do not have an effect on this cycle
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		total_loss += loss.item() * inputs.size(0)

	print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(train_loader.dataset):.4f}')

# Evaluate the Model
model.eval()

# Containers for all predictions and labels
all_preds = []
all_labels = []

# No gradient needed for evaluation
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs > 0.5).float()  # Apply threshold to get binary predictions

        # Store predictions and labels
        all_preds.extend(predicted.numpy())
        all_labels.extend(labels.numpy())

# Convert lists to numpy arrays for metrics calculation
all_preds = np.array(all_preds).flatten()
all_labels = np.array(all_labels).flatten()

# Calculate metrics
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

print("Precision:", precision_score(all_labels, all_preds))
print("Recall:", recall_score(all_labels, all_preds))
print("F1 Score:", f1_score(all_labels, all_preds))
print("ROC AUC:", roc_auc_score(all_labels, all_preds))

# Calculate accuracy
accuracy = np.mean(all_labels == all_preds)
print(f'Accuracy: {accuracy:.4f}')