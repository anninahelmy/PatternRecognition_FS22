# 2a - In this exercise you should aim to improve the recognition rate on the MNIST dataset using SVM.
# SVM - Use the provided training set to build your SVM. Apply the trained SVM to classify the test set.
# Investigate at least two different kernels and optimize the SVM parameters by means of cross-validation.


# The main idea is that based on the labeled data (training data) the algorithm tries to find the optimal hyperplane
# which can be used to classify new data points. In two dimensions the hyperplane is a simple line.
#SVM learns similarities.


#finding the optimal hyperplane: intuitively, the best line is the line that is far away from both classes.
#to have optimal solution, maximize the margin in both ways.


#basic steps:
# 1. select hyperplanes which separates data with no points between them
# 2. maximize the margin
# 3. average line will be the decision boundary.
import pandas as pd
import csv
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as matplot
import seaborn as sb


train = pd.read_csv("dataset01/mnist_train.csv")
test = pd.read_csv("dataset01/mnist_test.csv")

X_train = train.iloc[:500, 1:].values #samples
y_train = train.iloc[:500, 0].values #labels

X_test = test.iloc[:, 1:].values
y_test = test.iloc[:, 0].values

'''
The fit method of SVC class is called to train the algorithm on the training data, which is passed as a parameter to the fit method.
Regularization Parameter C tells the SVM optimization how much you want to avoid miss classifying each training example.
if the C is higher, optimization will choose smaller margin hyperplane, so training data miss classification rate will be lower
parameter Gamme: gamma parameter defines how far the influence of a single training example reaches. High Gamma will only consider points close
to the plausible hyperplane. Low Gamma will consider points at greater distance.
Margin: higher margin results better model, so better classifcation (or prediction). Margin should always be maximized.
'''

svc = SVC(kernel="linear",  C=1.0)
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print('Model accuracy score with linear kernel and C = 1.0: {0:0.4f}'. format(accuracy_score(y_test, pred))) #default hyperparameters: C = 1.0, kernel= rbf, gamma = auto


svc = SVC(kernel="linear",  C=100)
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print('Model accuracy score with linear kernel and C = 100: {0:0.4f}'. format(accuracy_score(y_test, pred))) #default hyperparameters: C = 1.0, kernel= rbf, gamma = auto


svc = SVC(kernel="linear",  C=1000)
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print('Model accuracy score with linear kernel and C = 1000: {0:0.4f}'. format(accuracy_score(y_test, pred))) #default hyperparameters: C = 1.0, kernel= rbf, gamma = auto

svc = SVC(kernel="linear",  C=10000)
svc.fit(X_train, y_train)
pred = svc.predict(X_test)
print('Model accuracy score with linear kernel and C = 10000: {0:0.4f}'. format(accuracy_score(y_test, pred))) #default hyperparameters: C = 1.0, kernel= rbf, gamma = auto

y_pred = svc.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred))) #training and test set accuarcy comparable


cm = confusion_matrix(y_test, pred)
matplot.subplots(figsize=(10, 6))
sb.heatmap(cm, annot = True, fmt = 'g')
matplot.xlabel("Predicted")
matplot.ylabel("Actual")
matplot.title("Heatmap Predicted and Actual")
matplot.show()


# with other kernels we can solve the problem if we have non linear data and there is no good way to separate our data
# with a linear hyperplane -- add another dimension
# instantiate classifier with default hyperparameters
svc=SVC()
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with default hyperparameters: {0:0.4f}'. format(accuracy_score(y_test, y_pred))) #default hyperparameters: C = 1.0, kernel= rbf, gamma = auto


svc=SVC(C=100.0)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with rbf kernel and C=100.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# instantiate classifier with rbf kernel and C=1000
svc=SVC(C=1000.0)
svc.fit(X_train,y_train)
y_pred=svc.predict(X_test)
print('Model accuracy score with rbf kernel and C=1000.0 : {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


# k-fold cross-validatioN:
#split training samples into K independent parts and use each part once for testing, compute the average accuracy
#experiment with different values for the system aprameter and choose the one that achieves the besta verage accuracy
#leave-one-out method: K equals the number of trianing samples, this method is particularly interesting for small data sets
