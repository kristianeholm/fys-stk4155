

import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error



def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4


datapoints = 500
x = np.sort(np.random.uniform(0, 1, datapoints))
y = np.sort(np.random.uniform(0, 1, datapoints))


z = FrankeFunction(x, y) + 0.4* np.random.normal(size=len(x))

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



polynomial = 5
K = 10 #number of runnings
average_mse_train = np.zeros(polynomial)
average_mse_test = np.zeros(polynomial)

average_R2_train = np.zeros(polynomial)
average_R2_test = np.zeros(polynomial)

betas = np.zeros((40, polynomial))

nlambdas = 100
lambdas = np.logspace(-3, 3, nlambdas)

MSE_train_lamdasXpoly_ridge = np.zeros((nlambdas, 5))
MSE_test_lamdasXpoly_ridge = np.zeros((nlambdas, 5))

for k in range(K):

    for b in range(nlambdas):


        # loop over polynomial degrees and calculating MSE

        poly = np.linspace(1, polynomial, polynomial)  # list of degress of polynomial we are running, not zero

        for i in range(1, polynomial + 1):

            ##create the design matrix with degree i
            X = create_X(x, y, i)

            # instead of splitting x and y data seperatly, we seperate the design matrix
            X_train, X_test, z_train, z_test = train_test_split(X, z, test_size=0.2)

            ##scaling
            # average of each column
            col_meansX = np.mean(X_train, axis=0)
            X_train = X_train - col_meansX

            col_means_z = np.mean(z_train, axis=0)
            z_train = z_train - col_means_z

            Xtrain = X_train  # [:,1:] #removing the first column, ie the intercept

            beta_ridge = (np.linalg.pinv(
                (Xtrain.T @ Xtrain + (lambdas[b] * np.identity(np.shape(X)[1])))) @ Xtrain.T) @ z_train

            beta = beta_ridge

            # Model prediction, we need also to transform our data set used for the prediction.
            ztildeTrain = Xtrain @ beta
            X_test = X_test - col_meansX  # Use mean from training data
            ztilde = X_test @ beta
            ztilde = ztilde + col_means_z

            mse_manuel = MSE(z_test, ztilde)
            mse_train = MSE(z_train, ztildeTrain)

            average_mse_train[i - 1] += mse_train
            average_mse_test[i - 1] += mse_manuel

            MSE_train_lamdasXpoly_ridge[b, i - 1] += mse_train
            MSE_test_lamdasXpoly_ridge[b, i - 1] += mse_manuel

MSE_train_lamdasXpoly_ridge /= k
#print(MSE_train_lamdasXpoly_ridge)

P, L = X,Y = np.meshgrid(np.array([1, 2, 3, 4, 5]), lambdas)

print(np.shape(P), np.shape(L), np.shape(MSE_train_lamdasXpoly_ridge))

ax = plt.axes(projection='3d')
ax.plot_surface(P, L, MSE_train_lamdasXpoly_ridge, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_title('ٱلْحَمْدُ لِلَّٰ');
ax.set_xlabel('poly degree')
ax.set_ylabel('lambda')
ax.set_zlabel('MSE Ridge');
plt.show()



