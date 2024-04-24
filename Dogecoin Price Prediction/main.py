import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('Dogecoin Price Prediction/DOGE-USD.csv')

df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
df.set_index('Date', inplace=True)
df.isnull().any()
df = df.dropna()

df['gap'] = (df['High'] - df['Low']) * df['Volume']
df['y'] = df['High'] / df['Volume']
df['z'] = df['Low'] / df['Volume']
df['a'] = df['High'] / df['Low']
df['b'] = (df['High'] / df['Low']) * df['Volume']
abs(df.corr()['Close']).sort_values(ascending=False)

data = df[['Close', 'Volume', 'gap', 'a', 'b']]
X_train, X_test, y_train, y_test = train_test_split(data.drop('Close', axis=1), data['Close'], test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(model.score(X_test, y_test))
plt.figure(figsize=(20,7))
plt.plot(y_test.values, label="Actual", color='b')
plt.plot(y_pred, label="Predicted", color='r')
plt.title('Dogecoin Price Prediction')
plt.xlabel('Amount')
plt.ylabel('Close Price')
plt.show()