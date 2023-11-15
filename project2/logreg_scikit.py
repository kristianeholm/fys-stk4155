#logistic regression (WITH scikit learn)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import numpy as npimport 
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

#Function for making the design matrix
def create_design_matrix(x, polynomial_degree=1):
    X = pd.DataFrame()
    for i in range(1, polynomial_degree + 1):
        col_name = 'x^' + str(i)
        #X[col_name] = x**i
        X[col_name] = x[:, 0]**i  # Access the first column of x
    return X

#get the breast cancer dataset
cancer=load_breast_cancer()      #Download breast cancer dataset

x = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
y = (cancer.target == 1).astype(int)                 #Label array of 569 rows (0 for benign and 1 for malignant)

#design matrix with input from x
X = create_design_matrix(x)

# Split the data into training and test --> training 80%, test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data with standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#using scikits logic regression function to train the model 
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Predict on the test set
y_prediction = logreg.predict(X_test_scaled)

#Find the accuracy
accuracy = accuracy_score(y_test, y_prediction)
print("Test Accuracy with Scikit-Learn Logistic Regression):", accuracy)

