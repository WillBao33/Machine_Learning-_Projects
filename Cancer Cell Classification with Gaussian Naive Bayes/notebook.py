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
