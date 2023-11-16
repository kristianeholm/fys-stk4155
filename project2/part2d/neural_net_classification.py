import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Add the parent directory to sys.path
sys.path.append("..") 

from functions import sigmoid, sigmoid_derivative, relu, relu_derivative, relu_leaky, relu_leaky_derivative
from metrics import MSE, R2, accuracy
from neural_network import NeuralNetwork

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

#design matrix
X = create_design_matrix(x)

# Split data in train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
#X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

################
num_epochs = 10000
minibatches = 5
learning_rate = 0.0005
lmbd = 1.0

network = NeuralNetwork(X_train_scaled.shape[1], 'class', 'sigmoid', cost_function='accuracy', minibatches=minibatches, epochs = num_epochs, eta=learning_rate, lmbd=lmbd)         
network.add_layer(12)
network.add_layer(12)
network.add_layer(1)    
     
network.train(X_train_scaled, y_train, data_val=X_test_scaled, target_val=y_test) 

test_pred = network.predict(X_test_scaled)
cost = accuracy(test_pred, y_test)
print('Our neural network, accuracy: ' + str(cost))

####
plt.title('Classification of breast cancer data, sigmoid activation, accuracy cost')
plt.plot(range(num_epochs), network.cost_train, label='Train error'); 
plt.plot(range(num_epochs), network.cost_test, label='Test error'); 
plt.xlabel('Number of iterations'); 
plt.ylabel('accuracy error')
plt.ylim(0.8, 1)
plt.legend()
plt.savefig('NN_classification_acc.pdf')
plt.show()
