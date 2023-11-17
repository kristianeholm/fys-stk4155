import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Add the parent directory to sys.path
sys.path.append("..") 

from functions import sigmoid, sigmoid_derivative, relu, relu_derivative, relu_leaky, relu_leaky_derivative
from metrics import MSE, R2
from neural_network import NeuralNetwork

from matplotlib import cm

def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4
    
def create_design_matrix(x, polynomial_degree=1):
    X = x#np.ones((len(x),1))
    for i in range(1, polynomial_degree + 1):
        X[i] = x**i
    return X
    
def create_X(x, y, n ):
	if len(x.shape) > 1:
		x = np.ravel(x)
		y = np.ravel(y)

	N = len(x)
	l = int((n+1)*(n+2)/2)		# Number of elements in beta
	X = np.ones((N,l))

	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			X[:,q+k] = (x**(i-k))*(y**k)

	return X
    
###################
noiseSpread = 0.1
datapoints = 400
n = datapoints
#x = 2 * np.random.rand(n)
#y = 4 + 3 * x + 2 * x**2 + np.random.normal(scale = 0.1, size = len(x))  # f(x) = a_0 + a_1x + a_2x^2
x = np.random.uniform(0, 1, datapoints)
y = np.random.uniform(0, 1, datapoints)
z = FrankeFunction(x, y) + np.random.normal(0,noiseSpread,size=len(x))

# Create design matrix, up to second degree
X = create_X(x, y, 1)

# Split data in train, validation and test set
X_train, X_test, z_train, z_test = train_test_split(X, z, test_size = 0.2, random_state=1)

# Scale the data
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
#X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

################
num_epochs = 10000 
minibatches = 5
learning_rate = 0.0001 
lmbd = 0 

network = NeuralNetwork(X_train_scaled.shape[1], 'regression', 'sigmoid', cost_function='MSE', minibatches=minibatches, epochs = num_epochs, eta=learning_rate, lmbd=lmbd)       
        
network.add_layer(30) 
network.add_layer(30) 
network.add_layer(1)    
     
network.train(X_train_scaled, z_train, data_val=X_test_scaled, target_val=z_test) 

test_pred = network.predict(X_test_scaled)
cost = MSE(test_pred, z_test)

print("MSE cost function on test set: ", cost)

#test_pred = network.predict(X_test_scaled)

polynomial=1

def surface_plot():
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    size = 100
    xx = np.sort(np.random.uniform(0, 1, size))
    yy = np.sort(np.random.uniform(0, 1, size))
    X = create_X(np.array([xx]), np.array([yy]), polynomial)
    Xgrid, Ygrid = np.meshgrid(xx, yy)
    Zgrid = network.predict(X)

    print(Xgrid.shape)
    print(Ygrid.shape)
    print(Zgrid.shape)
    print(X.shape)
    #Plot the surface.
    surf = ax.plot_surface(Xgrid, Ygrid, Zgrid, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    #X, Y = np.meshgrid(x, y)


    ax.scatter3D(x, y, z, color="green", label="data points")

    fig.colorbar(surf, ax=ax,
                 shrink=0.5,
                 aspect=9)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('franke(x, y)')
    ax.set_title(f"OLS surface degree {polynomial} ")
    ax.legend()
    plt.savefig("ols_3d_{}_{}_{}.pdf".format(polynomial, datapoints, K))
    plt.show()


#surface_plot()

#print 
X_test_for_plot = X_test#.to_numpy()
#plt.plot(x, y, 'r.', label='Data Points')
#plt.plot(X_test_for_plot[:,0], test_pred, 'g.', label='Model prediction')
#plt.xlabel(r'x')
#plt.ylabel(r'y')
#plt.title('Neural network regression')
#plt.legend()
#plt.savefig('NNregr_x_y.pdf')
#plt.show()
####
plt.plot(range(num_epochs), network.cost_train, label='Train error'); 
plt.plot(range(num_epochs), network.cost_test, label='Test error'); 
plt.xlabel('Number of iterations'); 
plt.ylabel('Mean squared error') 
plt.legend() 
plt.yscale('log') 
plt.savefig('NNregr_errors.pdf')
plt.show()
