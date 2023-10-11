import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.utils import resample
from sklearn import linear_model


def FrankeFunction(x,y):
	term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
	term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
	term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
	term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
	return term1 + term2 + term3 + term4

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



datapoints = 200
x = np.random.uniform(0, 1, datapoints)
y = np.random.uniform(0, 1, datapoints)


z = FrankeFunction(x, y) + 0.2* np.random.normal(size=len(x))


index = list(range(datapoints))
index = np.array(index)
index_train, index_test = train_test_split(index, test_size=0.3)


polynomial = 6
bootstraps = 50

#print(f"train index: {index_train}")
#print(f"test index: {index_test}")



error = np.zeros(polynomial)
bias = np.zeros(polynomial)
variance = np.zeros(polynomial)
polydegree = np.zeros(polynomial)


for p in range(1, polynomial + 1):

	y_pred = np.empty((len(index_test), bootstraps))

	for i in range(bootstraps):

		resample_indcies = resample(index_train, n_samples = int(len(index_train)*0.7) )
		#print(f"resample indecies: {resample_indcies}")

		x_, y_ = x[resample_indcies], y[resample_indcies]
		X = create_X(x_, y_, p)

		Xtest = create_X(x[index_test], y[index_test], p)  ##only depends on i


		clf = skl.LinearRegression().fit(X, z[resample_indcies])
		z_pred = clf.predict(Xtest)
		#print(f"predicted z: {z_pred}")

		y_pred[:, i] = z_pred


	#error[p] = np.mean(np.mean((z[index_test] - y_pred) ** 2, axis=1, keepdims=True))
	mse = np.zeros(bootstraps)
	for i in range(bootstraps):
		mse[i] = mean_squared_error(z[index_test], y_pred[:, i] )


	error[p-1] = np.mean(mse)
	bias[p-1] = np.mean((z[index_test] - np.mean(y_pred, axis=1, keepdims=True)) ** 2)

	variance[p-1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
	polydegree[p-1] = p

plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.plot(polydegree, variance + bias, label='Bias Variance')
plt.xlabel("polynomial degree")
plt.ylabel("")
plt.title("50 bootstraps")
plt.legend()
plt.show()




##cross validation

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((polynomial, k))



error = np.zeros(polynomial)
bias = np.zeros(polynomial)
variance = np.zeros(polynomial)
polydegree = np.zeros(polynomial)

for p in range(1, polynomial + 1):
	j = 0
	y_pred = np.empty((int(datapoints/k), k))

	for train_inds, test_inds in kfold.split(x):

		xtrain = x[train_inds]
		ytrain = y[train_inds]

		#create design matrix
		X = create_X(xtrain, ytrain, p)

		ztrain = z[train_inds]

		clf = skl.LinearRegression().fit(X, ztrain)

		xtest = x[test_inds]
		ytest = y[test_inds]
		#create test design matrix
		Xtest= create_X(xtest, ytest, p)

		ztest = z[test_inds]

		z_pred = clf.predict(Xtest)

		y_pred[:, j] = z_pred

		j += 1

	#print(y_pred)
	mse = np.zeros(k)
	for i in range(k):
		mse[i] = mean_squared_error(ztest, y_pred[:, i] )


	error[p-1] = np.mean(mse)
	bias[p-1] = np.mean((z[index_test] - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
	variance[p-1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
	polydegree[p-1] = p



plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.plot(polydegree, variance + bias, label='Bias + Variance')
plt.xlabel("polynomial degree")
plt.ylabel("")
plt.title(f"{k}-fold cross validation")
plt.legend()
plt.show()


##bias viariance cross variance ridge and lasso


k = 5
kfold = KFold(n_splits = k)

# Perform the cross-validation to estimate MSE
scores_KFold = np.zeros((polynomial, k))



error = np.zeros(polynomial)
bias = np.zeros(polynomial)
variance = np.zeros(polynomial)
polydegree = np.zeros(polynomial)
lamda = 0.05
for p in range(1, polynomial + 1):
	j = 0
	y_pred = np.empty((int(datapoints/k), k))

	for train_inds, test_inds in kfold.split(x):

		xtrain = x[train_inds]
		ytrain = y[train_inds]

		#create design matrix
		X = create_X(xtrain, ytrain, p)

		ztrain = z[train_inds]

		clf = skl.LinearRegression().fit(X, ztrain)


		##ridge and lasso only differ with few lines, for ridge calculation comment out the lasso,
		##for lasso, comment out the ridge


		"""should only have 1 of the belows"""
		# ridge
		beta_ridge = (np.linalg.pinv(
			(X.T @ X + (lamda * np.identity(np.shape(X)[1])))) @ X.T) @ ztrain

		ztildeTrain = X @ beta_ridge


		##lasso,
		#RegLasso = linear_model.Lasso(lamda)
		#clf = RegLasso.fit(X, ztrain)



		#this part is same for both
		xtest = x[test_inds]
		ytest = y[test_inds]
		#create test design matrix
		Xtest= create_X(xtest, ytest, p)
		ztest = z[test_inds]


		##only have one of these, the other should be commented out
		z_pred = Xtest @ beta_ridge #ridge
		#z_pred = clf.predict(Xtest) #lasso

		y_pred[:, j] = z_pred

		j += 1

	#print(y_pred)
	mse = np.zeros(k)
	for i in range(k):
		mse[i] = mean_squared_error(ztest, y_pred[:, i] )


	error[p-1] = np.mean(mse)
	bias[p-1] = np.mean((z[index_test] - np.mean(y_pred, axis=1, keepdims=True)) ** 2)
	variance[p-1] = np.mean(np.var(y_pred, axis=1, keepdims=True))
	polydegree[p-1] = p



plt.plot(polydegree, error, label='Error')
plt.plot(polydegree, bias, label='bias')
plt.plot(polydegree, variance, label='Variance')
plt.plot(polydegree, variance + bias, label='Bias + Variance')
plt.xlabel("polynomial degree")
plt.ylabel("")
plt.title(f"{k}-fold cross validation, Ridge l = {lamda}")
#plt.title(f"{k}-fold cross validation, Lasso l = {lamda}")
plt.legend()
plt.show()

