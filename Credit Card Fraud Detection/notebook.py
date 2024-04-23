#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec
# %%
df = pd.read_csv('Credit Card Fraud Detection/creditcard.csv')
df['Class'].head()
# %%
df.info()
# %%
df.shape
# %%
df.describe()
# %%
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)
print(f'Fraud Cases: {len(fraud)}')
print(f'Valid Cases: {len(valid)}')
# %%
print('Amount details of fraudulent transaction')
fraud.Amount.describe()
# %%
print('Amount details of valid transaction')
valid.Amount.describe()
# %%
corrmat = df.corr()
fig = plt.figure(figsize=(12, 9))
sns.heatmap(corrmat, vmax = 0.8, square=True)
plt.show()
# %%
X = df.drop(['Class'], axis=1)
y = df['Class']
print(X.shape)
print(y.shape)
X = X.values
y = y.values
# %%
from sklearn.model_selection import train_test_split
# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)
# %%
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
# %%
n_outliers = len(fraud)
n_errors = (y_pred != y_test).sum()
print('The model used is Random Forest Classifier')

acc = accuracy_score(y_test, y_pred)
print(f'The accuracy is {acc}')

prec = precision_score(y_test, y_pred)
print(f'The precision is {prec}')

rec = recall_score(y_test, y_pred)
print(f'The recall is {rec}')

f1 = f1_score(y_test, y_pred)
print(f'The F1-Score is {f1}')

MCC = matthews_corrcoef(y_test, y_pred)
print(f'The Matthews correlation coefficient is {MCC}')
# %%
LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
# %%
