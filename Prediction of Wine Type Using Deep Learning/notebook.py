#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# %%
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep =';')
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep =';')
# %%
fig, ax = plt.subplots(1, 2)
ax[0].hist(red.alcohol, 10, facecolor='red', alpha=0.5, label="Red wine")
ax[1].hist(white.alcohol, 10, facecolor='white', ec="black", lw=0.5, alpha=0.5, label="White wine")
fig.subplots_adjust(left = 0, right=1, bottom=0, top=0.5, hspace=0.05, wspace=1)

ax[0].set_ylim([0, 1000])
ax[0].set_xlabel("Alcohol in % Vol")
ax[0].set_ylabel("Frequency")
ax[1].set_ylim([0, 1000])
ax[1].set_xlabel("Alcohol in % Vol")
ax[1].set_ylabel("Frequency")

fig.suptitle("Distribution of Alcohol in % Vol")
plt.show()
# %%
red['type'] = 1
white['type'] = 0

wines = red.append(white, ignore_index=True)
wines.head()
# %%
X = wines.iloc[:, 0:11]
y = np.ravel(wines.type)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=45)
# %%
X_train.shape
type(X_train.values)
# %%
import torch
import torch.nn as nn
# %%
model = nn.Sequential(
    nn.Linear(11, 12),
    nn.ReLU(),
    nn.Linear(12, 9),
    nn.ReLU(),
    nn.Linear(9, 1),
    nn.Sigmoid()
)
model
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.BCELoss()
#%%
def binary_accuracy(y_pred, y_true):
    """Computes binary accuracy"""
    y_pred_tag = torch.round(y_pred)  # Directly round the predictions since they are already probabilities
    correct_results_sum = (y_pred_tag == y_true).sum().float()
    acc = correct_results_sum / y_true.shape[0]
    acc = torch.round(acc * 100)
    return acc

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_train = y_train.view(-1, 1)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)
X_test = torch.tensor(X_test.values, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
#%%
#%%
epochs = 3
model.to(device)
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')