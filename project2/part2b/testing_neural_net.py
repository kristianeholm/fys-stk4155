import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to sys.path
sys.path.append("..")

from metrics import MSE, R2
from neural_network import NeuralNetwork

def create_design_matrix(x, polynomial_degree=1):
    X = pd.DataFrame()
    for i in range(1, polynomial_degree + 1):
        col_name = 'x^' + str(i)
        X[col_name] = x**i
    return X

###################

n = 400
x = 2 * np.random.rand(n)
y = 4 + 3 * x + 2 * x**2 + np.random.normal(scale = 0.1, size = len(x))  # f(x) = a_0 + a_1x + a_2x^2

# Create design matrix, up to second degree
X = create_design_matrix(x, 1)

# Split data in train, validation and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################
num_epochs = 10000 
minibatches = 5 
learning_rate = 0.0001 
lmbd = 0 

network = NeuralNetwork(X_train_scaled.shape[1], 'regression', 'sigmoid', cost_function='MSE', minibatches=minibatches, epochs = num_epochs, eta=learning_rate, lmbd=lmbd)         
network.add_layer(12)
network.add_layer(12)
network.add_layer(1)
network.train(X_train_scaled, y_train, data_val=X_test_scaled, target_val=y_test)

test_pred = network.predict(X_test_scaled)

X_test_for_plot = X_test.to_numpy()
plt.plot(x, y, 'r.', label='Data Points')
plt.plot(X_test_for_plot[:,0], test_pred, 'g.', label='Model prediction')
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.title('Neural network regression')
plt.legend()
plt.savefig('NNregr_x_y.pdf')
plt.show()
####
plt.plot(range(num_epochs), network.cost_train, label='Train cost'); 
plt.plot(range(num_epochs), network.cost_test, label='Validation cost'); 
plt.xlabel('Number of iterations'); 
plt.ylabel('Mean squared error') 
plt.legend() 
plt.yscale('log') 
plt.savefig('NNregr_errors.pdf')
plt.show()
