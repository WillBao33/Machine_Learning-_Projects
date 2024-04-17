#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from tqdm.notebook import tqdm
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score as ras

import warnings
warnings.filterwarnings("ignore")
# %%
df=pd.read_csv("Parkinson Disease Prediction using ML/data.csv")
df.head()
# %%
df.shape
# %%
df.info()
# %%
df.describe()
# %%
df = df.groupby('id').mean().reset_index()
df.head()
# %%
df.drop('id', axis=1, inplace=True)
df.head()
# %%
columns = list(df.columns)
columns
# %%
for col in columns:
    if col == 'class':
        continue
    filtered_columns = [col]
    for col1 in df.columns:
        if ((col == col1) | (col == 'class')):
            continue

        val = df[col].corr(df[col1])
        if val > 0.7:
            columns.remove(col1)
        else:
            filtered_columns.append(col1)

    df = df[filtered_columns]

df.shape
# %%
X = df.drop('class', axis=1)
X_norm = MinMaxScaler().fit_transform(X)
selector = SelectKBest(chi2, k=30)
selector.fit(X_norm, df['class'])
filtered_columns = selector.get_support()
filtered_data = X.loc[:, filtered_columns]
filtered_data['class'] = df['class']
df = filtered_data
df.shape
# %%
df.head()
# %%
x = df['class'].value_counts()
plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
plt.show()
# %%
features = df.drop('class', axis=1)
target = df['class']

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=10)
X_train.shape, X_test.shape
# %%
ros = RandomOverSampler(sampling_strategy='minority', random_state=0)
X, y = ros.fit_resample(X_train, y_train)
X.shape, y.shape
# %%
models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf', probability=True)]

for i in range(len(models)):
    models[i].fit(X, y)

    print(f'Model: {models[i]} : ')
    train_preds = models[i].predict_proba(X)[:, 1]
    print('Training AUC: ', ras(y, train_preds))
    val_preds = models[i].predict_proba(X_test)[:, 1]
    print('Validation AUC: ', ras(y_test, val_preds))
    print()
# %%
plt.figure(figsize=(10,7))
sns.heatmap(metrics.confusion_matrix(y_test, models[0].predict(X_test)), annot=True)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# %%
print(metrics.classification_report(y_test, models[0].predict(X_test)))
# %%
