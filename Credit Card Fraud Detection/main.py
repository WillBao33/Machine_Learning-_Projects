import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Credit Card Fraud Detection/creditcard.csv')
fraud = df[df['Class'] == 1]
valid = df[df['Class'] == 0]
X = df.drop(['Class'], axis=1)
y = df['Class']
X = X.values
y = y.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

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

LABELS = ['Normal', 'Fraud']
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt='d')
plt.title('Confusion matrix')
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()