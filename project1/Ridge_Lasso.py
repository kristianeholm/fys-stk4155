

import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from sklearn import linear_model

#akward numpy


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
x = np.random.uniform(0, 1, datapoints)
y = np.random.uniform(0, 1, datapoints)


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



polynomial = 10
K = 1 #number of runnings

betas = np.zeros((40, polynomial))

##lambdas only valid for lasso from 0.01
nlambdas = 300
lambdas = np.logspace(-1, 1, nlambdas)


MSE_train_lamdasXpoly_ridge = np.zeros((nlambdas, polynomial))
MSE_test_lamdasXpoly_ridge = np.zeros((nlambdas, polynomial))

MSE_train_lamdasXpoly_lasso = np.zeros((nlambdas, polynomial))
MSE_test_lamdasXpoly_lasso = np.zeros((nlambdas, polynomial))

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
        col_meansX = 0  # np.mean(X_train, axis=0)
        X_train = X_train - col_meansX

        col_means_z = 0  # np.mean(z_train, axis=0)
        z_train = z_train - col_means_z

        Xtrain = X_train  # [:,1:] #removing the first column, ie the intercept

        # ridge
        beta_ridge = (np.linalg.pinv(
            (Xtrain.T @ Xtrain + (lambdas[b] * np.identity(np.shape(X)[1])))) @ Xtrain.T) @ z_train

        ztildeTrain = Xtrain @ beta_ridge

        ztilde = X_test @ beta_ridge
        ztilde = ztilde

        mse_manuel = MSE(z_test, ztilde)
        mse_train = MSE(z_train, ztildeTrain)

        MSE_train_lamdasXpoly_ridge[b, i - 1] = mse_train
        MSE_test_lamdasXpoly_ridge[b, i - 1] = mse_manuel

        # lasso
        RegLasso = linear_model.Lasso(lambdas[b])
        RegLasso.fit(X_train, z_train)

        MSE_lasso_train = MSE(z_train, RegLasso.predict(X_train))
        MSE_lasso_test = MSE(z_test, RegLasso.predict(X_test))

        MSE_train_lamdasXpoly_lasso[b, i - 1] = MSE_lasso_train
        MSE_test_lamdasXpoly_lasso[b, i - 1] = MSE_lasso_test



P, L = X,Y = np.meshgrid(np.array(list(range(1,polynomial+1))), lambdas)

print(np.shape(P), np.shape(L), np.shape(MSE_train_lamdasXpoly_ridge))

fig = plt.figure(figsize=(5, 3.5))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(P, L, MSE_train_lamdasXpoly_ridge, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
fig.colorbar(surf, ax=ax,
				 shrink=0.5,
				 aspect=9)

ax.set_title(' Ridge train error');
ax.set_xlabel('poly degree')
ax.set_ylabel('lambda')
ax.set_zlabel('MSE Ridge');

plt.show()


fig = plt.figure(figsize=(5, 3.5))
ax = plt.axes(projection='3d')
ax.plot_surface(P, L, MSE_test_lamdasXpoly_ridge, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

fig.colorbar(surf, ax=ax,
				 shrink=0.5,
				 aspect=9)

ax.set_title('Ridge test error');
ax.set_xlabel('poly degree')
ax.set_ylabel('lambda')
ax.set_zlabel('MSE Ridge');

plt.show()


#lasso plot

im = plt.pcolormesh(P, L, MSE_train_lamdasXpoly_lasso)  # drawing the function
plt.xlabel("polynomial")
plt.ylabel("lambda")
# adding the Contour lines with labels
plt.colorbar(im)  # adding the colobar on the right
# latex fashion title
plt.title("MSE lasso train")
plt.show()




