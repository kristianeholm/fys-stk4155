import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Add the parent directory to sys.path
sys.path.append("..") 

from metrics import MSE, R2
from neural_network import NeuralNetwork

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

def create_X(x, y):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)
    N = len(x)
    X = np.ones((N,2))
    X[:,0] = x
    X[:,1] = y
    return X

###################
noiseSpread = 0.1
datapoints = 300
x = np.random.uniform(0, 1, datapoints)
y = np.random.uniform(0, 1, datapoints)
z = FrankeFunction(x, y) + np.random.normal(0,noiseSpread,size=len(x))

# Create design matrix, up to 1 degree
X = create_X(x, y)

# Split data in train, validation and test set
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

################
num_epochs = 2000
minibatches = 5

# Learning rates to test
#eta_vals = [0.0001, 0.0005, 0.001, 0.005]
eta_vals = [ 0.00005, 0.0001, 0.0005, 0.001]
# Lambdas to test
lmbd_vals = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5, 10, 50]
##eta_vals = np.logspace(-5, 1, 7)
##lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

all_costs = {}
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
# grid search
for i, eta in enumerate(eta_vals):
    cost_rate = []
    for j, lmbd in enumerate(lmbd_vals):
        network = NeuralNetwork(X_train_scaled.shape[1], 'regression', 'relu', cost_function='MSE', minibatches=minibatches, epochs = num_epochs, eta=eta, lmbd=lmbd)         
        network.add_layer(12)
        network.add_layer(12)
        network.add_layer(1)
        network.train(X_train_scaled, z_train, data_val=X_test_scaled, target_val=z_test) 
        
        test_pred = network.predict(X_test_scaled)
        cost = MSE(test_pred, z_test)
        
        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("ReLU activation function on test set. MSE: ", cost)
        print()

        cost_rate.append(cost)
    all_costs[eta] = cost_rate
        
sns.set()
# Create dataframs for errors
df_cost = pd.DataFrame(all_costs)
df_cost.index = lmbd_vals

sns.heatmap(df_cost, annot=True, cmap="viridis")
plt.ylabel(r'Regularization parameter, $\lambda$')
plt.xlabel(r'Learning rate');
plt.title('NN ReLU activation tuning')
plt.savefig('heatmap_nn_train_franke_relu.pdf')
