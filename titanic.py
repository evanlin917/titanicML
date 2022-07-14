import numpy as np
import pandas as pd
import matplotlib.pyplot as pyplot
import sklearn

#using pandas to create a DataFrame to organize the data
df = pd.read_csv("titanic_train.csv")
df = df.drop("passenger_id", axis = 'columns')
df = df.drop("name", axis = 'columns')
df = df.drop("ticket", axis = 'columns')
df = df.drop("embarked", axis = 'columns')
df = df.drop("cabin", axis = 'columns')
df = df.drop("boat", axis = 'columns')
df = df.drop("body", axis = 'columns')
df = df.drop("home.dest", axis = 'columns')
df = df.dropna(subset = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare'])
print(df.head)

#creates a new column in the DataFrame with boolean values denoting if certain passengers are male or female
df['male'] = df['sex'] == 'male'
print(df.head)

#defining the feature matrix and creating a NumPy array to hold the data of the DataFrame
x = df[['pclass', 'male', 'age', 'sibsp', 'parch', 'fare']].values
print(x)

#defining the target and creating a NumPy array to hold the data of the DataFrame
y = df['survived'].values
print(y)

#importing the Logistic Regression model from Scikit-Learn
from sklearn.linear_model import LogisticRegression

#creating the logistic regression ML model
model = LogisticRegression()
model.fit(x,y)

#determining the coefficients surrounding the line-of-best-fit
print(model.coef_, model.intercept_)

#predicting if certain passengers will survive given inputs of class, male/female, age, siblings/spouses, parents/children, and fare
print(model.predict([[3, True, 38.0, 0, 0, 8.6625]])) #first passenger

#predicting the survivability of the first five passengers and comparing it to the real survivability rates
print(model.predict(x[:5]))
print(y[:5])

#creating an array of predicted y values
y_pred = model.predict(x)

#comparing to see if each prediction is correct and printing the number of correct predictions
y == y_pred
print((y == y_pred).sum())

#determining the percentage of accuracy employed by the model
y.shape[0] #gives the total number of data points in the set
print((y == y_pred).sum() / y.shape[0])
print(model.score(x, y)) #alternative way of getting the accuracy