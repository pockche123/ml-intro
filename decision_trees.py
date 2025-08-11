import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("X_train\n", X_train, "\n")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

print("Training the Decision Tree Classification model on the Training set\n")
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

print("\nPredicting a new result\n")
print(classifier.predict(sc.transform([[30,87000]])))

y_pred = classifier.predict(X_test)
print("\nPredicting the Test set results - left side is predictions while right is actual\n")
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

print("\nMaking the Confusion Matrix - ie testing the accuracy of the prediction\n")
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm,"\n")
print("Accuracy of the model")
print(accuracy_score(y_test, y_pred))