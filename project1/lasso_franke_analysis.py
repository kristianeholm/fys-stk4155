
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from sklearn.utils import resample
import sys

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

#if len(sys.argv) < 5:
#    sys.exit('Usage: project1.py <polynomial degree> <datapoints> <sigma^2 for random noise> <number of runs>')
#Above if we want to require parameters, however the below restures the old default values.
if len(sys.argv) < 6:
	polynomial = 12
	datapoints = 300
	noiseSpread = 0.1
	K = 1
	lambda_ridge = 0.01
else:
	polynomial = int(sys.argv[1])
	datapoints = int(sys.argv[2])
	noiseSpread = float(sys.argv[3])
	K = int(sys.argv[4])
	lambda_ridge = float(sys.argv[5])

#print('Number of runs {}'.format(K))

#polynomial = 5
#K = 1 #number of runnings

x = np.random.uniform(0, 1, datapoints)
y = np.random.uniform(0, 1, datapoints)


z = FrankeFunction(x, y) + np.random.normal(0,noiseSpread,size=len(x))

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


def plot_surface(x2, y2):
	
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')
	#x2 = y2 = np.arange(0, 1, 0.05)
	X, Y = np.meshgrid(x2, y2)
	#zs = np.array(fun(np.ravel(X), np.ravel(Y), beta, n))

	zs = np.array(FrankeFunction(np.ravel(X), np.ravel(Y)))
	Z = FrankeFunction(X, Y)

	ax.plot_surface(X, Y, Z, label="leastsquare surface")
	#ax.scatter3D(x, y, Z, color="green", label="data")
	##ax.scatter3D(xx, yy, ztilde, color="black", label="leastsquare")

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('f(x, y)')
	plt.savefig("ridge_surface_{}_{}_{}.pdf".format(polynomial, datapoints, K))
	plt.show()
#plot_surface(x,y)






average_mse_train = np.zeros(polynomial)
average_mse_test = np.zeros(polynomial)

average_R2_train = np.zeros(polynomial)
average_R2_test = np.zeros(polynomial)

betas = np.zeros((150, polynomial))



for k in range(K):

	# loop over polynomial degrees and calculating MSE R2 and individial beta values

	MSE_values_testxxx = np.zeros(polynomial)
	MSE_values_train = np.zeros(polynomial)
	#R2_values_test = np.zeros(polynomial)

	poly = np.linspace(1, polynomial, polynomial) #list of degress of polynomial we are running, not zero

	for i in range(1, polynomial + 1):

		##apply bootstrap here
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

		# train model with training data
		# beta = (np.linalg.inv((X_train.T @ X_train)) @ X_train.T) @ z_train
		# beta = (np.linalg.pinv((Xtrain.T @ Xtrain)) @ Xtrain.T) @ z_train
        
        # ridge
		beta = (np.linalg.pinv(
            (Xtrain.T @ Xtrain + (lambda_ridge * np.identity(np.shape(X)[1])))) @ Xtrain.T) @ z_train
        # lasso
        #(alpha=self.Lambda, max_iter=1000, normalize=False)
		RegLasso = linear_model.Lasso(alpha=lambda_ridge, max_iter=1000000)
        #, tol=1e-3)
		RegLasso.fit(X_train, z_train)
        
		#z_predict[:, beta]       = RegLasso.predict(X_test)

		mse_train = MSE(z_train, RegLasso.predict(X_train))
		mse_test = MSE(z_test, RegLasso.predict(X_test))


		##using skitlearn functions

		clf = skl.LinearRegression().fit(Xtrain, z_train)


		# Model prediction, we need also to transform our data set used for the prediction.
		ztildeTrain = Xtrain @ beta
		X_test = X_test - col_meansX  # Use mean from training data
		ztilde = X_test @ beta
		ztilde = ztilde + col_means_z

		zpredict_skl = clf.predict(X_test)
		zpredict_skl += col_means_z

		# The mean squared error and R2 score
		#mse_test = MSE(z_test, ztilde)
		#mse_train = MSE(z_train, ztildeTrain)
		#mse_test_sklr = MSE(z_test, zpredict_skl)


		r2 = R2(z_test, ztilde)
		r22 = R2(z_train, ztildeTrain)


		#MSE_values_train[i-1] = mse_train
		#MSE_values_test[i-1] = mse_manuel

		#R2_values_test[i-1] = r2
		"""print(f"poly {i} ")
		print(f"train mse: {mse_train} ")
		print(f"test mse: {mse_test} ")
		print(f"test mse sklr: {mse_test_sklr} ")
		print(f"betas {beta}")
		print(f"betas sklr : {clf.coef_}")"""


		average_mse_train[i-1] += mse_train
		average_mse_test[i-1] += mse_test

		average_R2_train[i-1] +=r22
		average_R2_test[i-1] +=r2


		##cathing the beta values
		for b in range(len(beta)):
			betas[b, i-1] += beta[b]

		#maxnumberofbeta = len(beta)
		#print(len(beta))






average_mse_train = average_mse_train/K
average_mse_test = average_mse_test/K
average_R2_train = average_R2_train/K
average_R2_test = average_R2_test/K

fig = plt.figure()
gs = fig.add_gridspec(3, hspace=0)
axs = gs.subplots(sharex=True, sharey=False)

fig.suptitle('Lasso model λ={}'.format(lambda_ridge))
axs[0].plot(poly[:], average_mse_train[:], label = " mean MSE train" )
axs[0].plot(poly[:], average_mse_test[:], label = "mean MSE test" )
axs[0].legend()

axs[1].plot(poly[:], average_R2_train, label = "mean R2 train" )
axs[1].plot(poly[:], average_R2_test, label = "mean R2 test")
axs[1].legend()

for b in range(polynomial):
	axs[2].plot(poly[:], betas[b, :]/K, label=f"B{b}")


fig.text(0.5, 0.04, 'Polynomial degree', ha='center', va='center')
plt.legend()

plt.savefig("lasso_model_{}_{}_{}_lambda{}.pdf".format(polynomial, datapoints, K, lambda_ridge))
plt.show()


def surface_plot():
	fig = plt.figure(figsize=(10, 7))
	ax = fig.add_subplot(111, projection='3d')
	size = 100
	xx = np.sort(np.random.uniform(0, 1, size))
	yy = np.sort(np.random.uniform(0, 1, size))

	Xgrid, Ygrid = np.meshgrid(xx, yy)

	ZGrid = np.zeros((size, size))
	for i in range(size):
		for j in range(size):
			x1 = Xgrid[0, i]
			y1 = Ygrid[j, 0]
			X = create_X(np.array([x1]), np.array([y1]), polynomial)
			X = X - col_meansX
			Z = X @ beta
			ZGrid[i, j] = Z[0] + col_means_z

			"""print(f"x point : {x1}")
			print(f"y point : {y1}")
			print(f"expected value: {FrankeFunction(x1, y1)}")
			print(f"calculated value : {Z + col_means_z}")
			print("-------")"""



	#Plot the surface.
	surf = ax.plot_surface(Xgrid, Ygrid, ZGrid, cmap=cm.coolwarm,
						   linewidth=0, antialiased=False)
	#X, Y = np.meshgrid(x, y)


	ax.scatter3D(x, y, z, color="green", label="data points")

	fig.colorbar(surf, ax=ax,
				 shrink=0.5,
				 aspect=9)

	ax.set_xlabel('x')
	ax.set_ylabel('y')
	ax.set_zlabel('franke(x, y)')
	ax.set_title("Ridge surface degree {polynomial} ")
	ax.legend()
	plt.savefig("lasso_3d_{}_{}_{}_lambda{}.pdf".format(polynomial, datapoints, K))
	plt.show()


surface_plot()