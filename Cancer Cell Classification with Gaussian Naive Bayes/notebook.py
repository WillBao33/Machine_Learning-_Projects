#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# %%
data = load_breast_cancer()
# %%
label_names = data['target_names']
labels = data['target']
feature_names = data['feature_names']
features = data['data']
# %%
print(label_names)
# %%
print(labels)
# %%
print(feature_names)
# %%
print(features)
# %%
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
# %%
model = GaussianNB()
model.fit(X_train, y_train)
# %%
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.2f}')
# %%
from sklearn.neighbors import KNeighborsClassifier
# %%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# %%
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()
# %%
df_label = pd.DataFrame(data.target, columns=['label'])
df_label.head()
# %%
sns.lmplot(x = 'mean radius', y = 'mean texture', data = pd.concat([df, df_label], axis=1), hue='label')
# %%
sns.lmplot(x = 'smoothness error', y = 'compactness error', data=pd.concat([df, df_label], axis = 1), hue = 'label')
# %%
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print(f'Test Accuracy: {knn.score(X_test, y_test): .2f}')
# %%
# cross validation
neighbors = []
cv_scores = []
from sklearn.model_selection import cross_val_score
# perform 10 fold cross validation
for k in range(1, 51, 2):
    neighbors.append(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv = 10, scoring='accuracy')
    cv_scores.append(scores.mean())
# %%
# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print(f'The optimal number of neighbors is % d ' % optimal_k)

plt.figure(figsize=(10, 6))
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neightbors K')
plt.ylabel('Misclassification Error')
plt.show()