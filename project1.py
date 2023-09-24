
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


def fun(x, y, beta, n):
	f = beta[0]
	for i in range(1,n+1):
		q = int((i)*(i+1)/2)
		for k in range(i+1):
			#print(q+k)
			f += beta[q+k] * (x**(i-k))*(y**k)
	return f


def plot_surface(x2, y2, beta, n):
	
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')
	#x2 = y2 = np.arange(0, 1, 0.05)
	X, Y = np.meshgrid(x2, y2)
	zs = np.array(fun(np.ravel(X), np.ravel(Y), beta, n))

	zs = np.array(FrankeFunction(np.ravel(X), np.ravel(Y)))
	Z = zs.reshape(X.shape)

	ax.plot_surface(X, Y, Z, label="leastsquare surface")
	ax.scatter3D(x, y, z, color="green", label="data")
	##ax.scatter3D(xx, yy, ztilde, color="black", label="leastsquare")

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('f(x, y)')
	plt.show()


#plot_surface(x,y,beta,5)


polynomial = 5
K = 10 #number of runnings
average_mse_train = np.zeros(polynomial)
average_mse_test = np.zeros(polynomial)

average_R2_train = np.zeros(polynomial)
average_R2_test = np.zeros(polynomial)

betas = np.zeros((40, polynomial))

nlambdas = 100
lambdas = np.logspace(-4, 4, nlambdas)

for k in range(K):

	# loop over polynomial degrees and calculating MSE R2 and individial beta values

	MSE_values_test = np.zeros(polynomial)
	MSE_values_train = np.zeros(polynomial)
	R2_values_test = np.zeros(polynomial)
	poly = np.linspace(1, polynomial, polynomial) #list of degress of polynomial we are running, not zero



	for i in range(1, polynomial + 1):

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

		# train model with train data
		# beta = (np.linalg.inv((X_train.T @ X_train)) @ X_train.T) @ z_train
		beta = (np.linalg.pinv((Xtrain.T @ Xtrain)) @ Xtrain.T) @ z_train

		beta_ridge = (np.linalg.pinv((Xtrain.T @ Xtrain  +  (0.5 * np.identity(np.shape(X)[1])) )) @ Xtrain.T) @ z_train


		# Model prediction, we need also to transform our data set used for the prediction.
		ztildeTrain = Xtrain @ beta
		X_test = X_test - col_meansX  # Use mean from training data
		ztilde = X_test @ beta
		ztilde = ztilde + col_means_z

		# scaler = StandardScaler()
		# scaler.fit(X_train)
		# scaled data
		# X_train_scaled = scaler.transform(X_train)
		# X_test_scaled = scaler.transform(X_test)

		# clf = skl.LinearRegression().fit(X_train, z_train)
		# clf = skl.LinearRegression().fit(X_train_scaled, z_train)

		# The mean squared error and R2 score
		# mse = mean_squared_error(clf.predict(X_test), z_test)
		mse_manuel = MSE(z_test, ztilde)
		mse_train = MSE(z_train, ztildeTrain)


		r2 = R2(z_test, ztilde)
		r22 = R2(z_train, ztildeTrain)


		#MSE_values_train[i-1] = mse_train
		#MSE_values_test[i-1] = mse_manuel

		#R2_values_test[i-1] = r2

		average_mse_train[i-1] += mse_train
		average_mse_test[i-1] += mse_manuel

		average_R2_train[i-1] +=r22
		average_R2_test[i-1] +=r2


		##cathing the beta values
		for b in range(len(beta)):
			betas[b, i-1] += beta[b]

		#maxnumberofbeta = len(beta)
		#print(len(beta))








fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)

fig.suptitle('SUUUUUPEEEEEERR')
axs[0].plot(poly[:], average_mse_train[:]/k, label = " mean MSE train" )
axs[0].plot(poly[:], average_mse_test[:]/k, label = "mean MSE test" )
axs[0].legend()

axs[1].plot(poly[:], average_R2_train[:]/k, label = "mean R2 train" )
axs[1].plot(poly[:], average_R2_test[:]/k, label = "mean R2 test")
axs[1].legend()

for b in range(polynomial):
	axs[2].plot(poly[:], betas[b, :]/K, label=f"B{b}")


fig.text(0.5, 0.04, 'Polynomial degree', ha='center', va='center')
plt.legend()

plt.show()

