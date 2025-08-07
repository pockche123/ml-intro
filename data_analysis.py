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

print("Independent variable \n", X)

print("\n Dependent variable \n", y)


# 2. taking care of missing data 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# looks at all x but looks at age and salary column 1 to 3 (exclusve)
imputer.fit(X[:, 1:3])

# This has transformed the x where the missing values are now NAN
X[:, 1:3] = imputer.transform(X[:, 1:3])
print("\n Missing data \n", X, "\n")

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# 2. Encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# the first y index which is Country
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])],remainder='passthrough')
X  = np.array(ct.fit_transform(X))

print("The first three columns indicate their unique ids of 'France', 'Germany' and 'Spain' \n")
print(X, "\n")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(y)
print("For dependant variables, 0 corresponds to 'no' and 1 corresponds to '1'")
print(y, "\n")


print("\nFeature Scaling - happens after splitting dataset into training set and test set")
print("\n Feature scaling gets mean and standard deviation including the ones in the test set")


print("\n\n Let's split the dataset into Training set and Test set")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)    
print("\nX_train - represents 8 random customers taken from the dataset\n", X_train, "\n")
print("X_test- the last 2 columns are age and salary\n", X_test, "\n")
print("y_train - this is the 8 train purchase data\n", y_train, "\n")
print("y_test\n", y_test, "\n")


print("\n\nFeature Scaling - happens after splitting dataset into training set and test set")