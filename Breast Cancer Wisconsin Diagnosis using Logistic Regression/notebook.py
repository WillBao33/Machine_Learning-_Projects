#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# %%
df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header=None)
df.head()
# %%
df.info()
# %%
X = df.loc[:, 2:].values
y = df.loc[:, 1].values
y = [1 if i == 'M' else 0 for i in y]
y = np.array(y, dtype=float)
# %%
X.shape, y.shape
# %%
# normalize the data
X = (X-np.min(X))/(np.max(X)-np.min(X))
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train.shape, X_test.shape
# %%
X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T
#%%
y_train = np.array(y_train, dtype=float)
y_test = np.array(y_test, dtype=float)
# %%
def initialize_weights_and_bias(dimension):
    w = np.full((dimension, 1), 0.01) # w is size of (num_features, 1) and all values are 0.01, which is (30, 1) in this case
    b = 0.0
    return w, b
# %%
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head
# %%
def forward_backward_propagation(w, b, x_train, y_train):
    # forward propapation
    z = np.dot(w.T, x_train) + b  # z = w.T*x + b -> z is a scalar
    y_head = sigmoid(z) # y_head is a scalar
    loss =  -y_train*np.log(y_head) - (1-y_train)*np.log(1-y_head) # negative log likelihood
    cost = (np.sum(loss))/x_train.shape[1] # 455 samples in each row (30, 455)

    # backward propagation
    derivative_weight = (np.dot(x_train, ((y_head-y_train).T)))/x_train.shape[1] # dL/dw = dL/dy_head * dy_head/dz * dz/dw
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1] # dL/db = dL/dy_head * dy_head/dz * dz/db
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients
# %%
def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i: %f" %(i, cost))

    parameters = {"weight": w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
    return parameters, gradients, cost_list
# %%
def predict(w, b, x_test):
    z = sigmoid(np.dot(w.T, X_test)+b)
    y_pred = np.zeros((1, x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0, i] <= 0.5:
            y_pred[0, i] = 0
        else:
            y_pred[0, i] = 1

    return y_pred
# %%
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0]
    w, b = initialize_weights_and_bias(dimension)
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)

    y_pred_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_pred_train = predict(parameters['weight'], parameters['bias'], x_train)

    print("Train accuracy: {} %".format(100 - np.mean(np.abs(y_pred_train - y_train)) * 100)) # 100 - mean of absolute values of differences between y_pred_train and y_train
    print("Test accuracyL {} %".format(100 - np.mean(np.abs(y_pred_test - y_test)) * 100))
# %%
logistic_regression(X_train, y_train, X_test, y_test, learning_rate=2, num_iterations=100)
# %%
model = LogisticRegression(random_state=42, max_iter=100)
print("Train accuracy: {}".format(model.fit(X_train.T, y_train.T).score(X_train.T, y_train.T)))
print("Test accuracy: {}".format(model.fit(X_train.T, y_train.T).score(X_test.T, y_test.T)))
# %%
