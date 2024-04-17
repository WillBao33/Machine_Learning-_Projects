import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')

red['type'] = 1
white['type'] = 0
wines = red.append(white, ignore_index=True)

X = wines.iloc[:, 0:11]
y = np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=45)

model = nn.Sequential(
    nn.Linear(11, 12),
    nn.ReLU(),
    nn.Linear(12, 9),
    nn.ReLU(),
    nn.Linear(9, 1),
    nn.Sigmoid()
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

epochs = 3
model.to(device)
for epoch in range(epochs):
    model.train()
    total_correct = 0
    total_samples = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        y_pred = model(inputs)

        # calculate the accuracy
        predictions = (y_pred > 0.5).float()
        correct = (predictions == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)

        # calculate the loss
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = total_correct / total_samples

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
model.eval()
test_correct = 0
test_samples = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        test_pred = model(inputs)
        test_predictions = (test_pred > 0.5).float()
        test_correct += (test_predictions == labels).sum().item()
        test_samples += labels.size(0)

accuracy = test_correct / test_samples
print(f'Test accuracy: {accuracy: .4f}')