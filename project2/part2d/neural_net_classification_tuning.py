import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# Add the parent directory to sys.path
sys.path.append("..") 

#from functions import sigmoid, sigmoid_derivative, relu, relu_derivative, relu_leaky, relu_leaky_derivative
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

# Learning rates to test
eta_vals = [ 0.0001, 0.0005, 0.001, 0.005, 0.01]
# Lambdas to test
lmbd_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10, 50]

all_costs = {}
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
# grid search
for i, eta in enumerate(eta_vals):
    cost_rate = []
    for j, lmbd in enumerate(lmbd_vals):
        network = NeuralNetwork(X_train_scaled.shape[1], 'class', 'leakyrelu', cost_function='accuracy', minibatches=minibatches, epochs = num_epochs, eta=eta, lmbd=lmbd)         
        network.add_layer(32)
        network.add_layer(32)
        network.add_layer(1)
        network.train(X_train_scaled, y_train, data_val=X_test_scaled, target_val=y_test) 
        
        test_pred = network.predict(X_test_scaled)
        cost = accuracy(test_pred, y_test)
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy cost function on test set: ", cost)
        print()

        cost_rate.append(cost)
    all_costs[eta] = cost_rate
        
sns.set()
# Create dataframs for errors
df_cost = pd.DataFrame(all_costs)
df_cost.index = lmbd_vals

sns.heatmap(df_cost, annot=True, cmap="viridis");
plt.ylabel(r'Regularization parameter, $\lambda$');
plt.xlabel(r'Learning rate');
plt.savefig('heatmap_nn_train_leaky.pdf')
