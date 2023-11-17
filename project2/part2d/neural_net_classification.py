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

    
cancer=load_breast_cancer()      #Download breast cancer dataset

X = cancer['data']
y = cancer['target']

# Split data in train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state=1)
# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

print('Number train data: ' + str(X_train.shape))
print('Number test data: ' + str(X_test.shape))

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################
num_epochs = 1500
minibatches = 5
learning_rate = 0.0001
lmbd = 1.0

network = NeuralNetwork(X_train.shape[1], 'class', 'sigmoid', cost_function='accuracy', minibatches=minibatches, epochs = num_epochs, eta=learning_rate, lmbd=lmbd)         
network.add_layer(32)
network.add_layer(32)
network.add_layer(1)    
     
network.train(X_train, y_train, data_val=X_test, target_val=y_test) 

test_pred = network.predict(X_test)
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
