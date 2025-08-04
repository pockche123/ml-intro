import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Your code here
dataset = pd.read_csv('Data.csv')
# iloc = index location at x, y -> we want all the x rows, for y its 0 to last index (excluding)

#  X is the independent variable 
X = dataset.iloc[:, :-1].values

#  y is the dependent variable - we are syaing if purchasing is affected by other factors
y = dataset.iloc[:, -1].values

print(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)