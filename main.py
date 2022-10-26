import pandas as pd

url = "./excel.csv"

manhattan = pd.read_csv(url)
manhattan.head()
print(manhattan)
corr_matrix = manhattan.corr()

print(corr_matrix["chest"].sort_values(ascending=False))

corr_df = pd.DataFrame(corr_matrix["chest"].sort_values(ascending=False))

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.bar(corr_df.index, corr_df["chest"])
plt.xticks(rotation=45)
plt.show()

from pandas.plotting import scatter_matrix

attributes = ['sex','age','weight','chest','neck','back','leg']

scatter_matrix(manhattan[attributes], figsize=(12, 8))
plt.show()

from sklearn.model_selection import train_test_split

x = manhattan[['sex','age','weight','neck','back','leg']]
y = manhattan[['chest']]

x_train, x_test, y_train, y_test = train_test_split(x,y, train_size = 0.7, test_size = 0.3)

from sklearn.linear_model import LinearRegression

mlr = LinearRegression()
mlr.fit(x_train, y_train)

plt.plot(mlr.predict(x_test[:100]), label="predict")
plt.plot(y_test[:100].values.reshape(-1, 1), label="real price")
plt.legend()
plt.show()

y_predict = mlr.predict(x_test)

plt.scatter(y_test, y_predict, alpha = 0.4)
plt.show()
print(mlr.score(x_train, y_train))