#%%
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import statsmodels.api as sm
from sklearn import preprocessing
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# %%
disease_df = pd.read_csv('Heart Disease Prediction Using Logistic Regression/framingham.csv')
disease_df.head()
# %%
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns={'male':'Sex_male'}, inplace=True)
# %%
disease_df.dropna(axis=0, inplace=True)
# %%
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())
# %%
X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])
X = preprocessing.StandardScaler().fit(X).transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print(f"Train: {X_train.shape}, {y_train.shape}")
print(f"Test: {X_test.shape}, {y_test.shape}")
# %%
plt.figure(figsize=(7,5))
sns.countplot(x='TenYearCHD',data=disease_df, palette='BuGn_r')
plt.show()
# %%
laste = disease_df['TenYearCHD'].plot()
plt.show()
# %%
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
# %%
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
# %%
cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm, columns = ['Predicted:0', 'Predicted:1'], index = ['Actual:0', 'Actual:1'])
plt.figure(figsize=(8,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.show()
print('Classification Report: \n', classification_report(y_test, y_pred))
# %%
