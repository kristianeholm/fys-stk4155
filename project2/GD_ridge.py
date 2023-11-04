#GD with ridge regression
from random import random, seed
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import sys

np.random.seed(89)

# Formula for MSE
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

# the number of datapoints
n = 200
x = 2*np.random.rand(n,1)
y = 4 + 3*x + 2*x**2 + np.random.randn(n,1) #f(x)=a_0+a_1x+a_2x^2

X = np.c_[np.ones((n,1)), x, x**2]
XT_X = X.T @ X

#Ridge parameter lambda
lmbda  = 0.2
Id = n*lmbda* np.eye(XT_X.shape[0])

beta_linreg = np.linalg.inv(XT_X+Id) @ X.T @ y
beta = np.random.randn(3,1)

eta = 0.2 #learning rate
Niterations = 1000

for iter in range(Niterations):
    gradients = 2.0/n*X.T @ (X @ (beta)-y)+2*lmbda*beta
    beta -= eta*gradients


ypredict = X @ beta
ypredict2 = X @ beta_linreg

# Calculate MSE 
mse_ridge = MSE(y, ypredict2)
print(f'MSE for Ridge Regression: {mse_ridge:.2f}')

plt.plot(x, ypredict, "b-", label=f'Gradient Descent')
plt.plot(x, y, 'ro', label='Data Points')
plt.axis([0, 2.0, 0, 15.0])
plt.xlabel(r'x')
plt.ylabel(r'y')
plt.title('Gradient Descent for Ridge Regression')
plt.legend()
plt.show()