#SGD with and without momentum + adam
from math import exp, sqrt
from random import random, seed
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad

np.random.seed(89)

# Formula for MSE
def MSE(y_data, y_model):
    n = np.size(y_model)
    return np.sum((y_data - y_model)**2) / n

def CostOLS(y, X, theta):
    return np.sum((y - X @ theta)**2)

n = 400 #datapoints
x = 2 * np.random.rand(n, 1)
y = 4 + 3 * x + 2 * x**2 + np.random.randn(n, 1)  # f(x) = a_0 + a_1x + a_2x^2

X = np.c_[np.ones((n, 1)), x, x**2]

theta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y

# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS, 2)

# number of epochs and minibatches
n_epochs = 50
minibatch_size = 20


t0, t1 = 10, 35  # learning schedule parameters

# learning schedule function
def learning_schedule(t):
    return t0 / (t + t1)


# Initialization of Theta for SGD without momentum
theta_sgd = np.random.randn(3, 1) #without 
theta_momentum = np.random.randn(3, 1) #with


delta = 1e-8  # adding the adagrad parameter to avoid division by zero

G = np.zeros(theta_sgd.shape) #no momentum
G_momentum = np.zeros(theta_momentum.shape) #momentum

beta = 0.9
beta2 = 0.999
iter = 0

# SGD without momentum + adam
for epoch in range(n_epochs):
    first_moment = 0.0
    second_moment = 0.0
    iter += 1

    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index + minibatch_size]
        yi = y[random_index:random_index + minibatch_size]

        gradients = (1.0/minibatch_size)*training_gradient(yi, xi, theta_sgd)

        # Computing moments first
        first_moment = beta*first_moment + (1-beta)*gradients
        second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
        first_term = first_moment/(1.0-beta**iter)
        second_term = second_moment/(1.0-beta2**iter)

        eta = learning_schedule(epoch * (n // minibatch_size) + i) #eta with learning schedule
        update = eta * first_term / (np.sqrt(second_term) + delta)
        theta_sgd -= update

#velocity for SGD with momentum
velocity = np.zeros((3, 1))
mu = 0.05  # momentum

# SGD with momentum + RMSprop
for epoch in range(n_epochs):
    first_moment = 0.0
    second_moment = 0.0
    iter += 1
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index + minibatch_size]
        yi = y[random_index:random_index + minibatch_size]

        gradients_momentum = (1.0 / minibatch_size) * training_gradient(yi, xi, theta_momentum)

        # Computing moments first
        first_moment = beta*first_moment + (1-beta)*gradients
        second_moment = beta2*second_moment+(1-beta2)*gradients*gradients
        first_term = first_moment/(1.0-beta**iter)
        second_term = second_moment/(1.0-beta2**iter)

        eta = learning_schedule(epoch * n // minibatch_size + i) #eta from learning schedule

        velocity = mu * velocity + eta * gradients_momentum
        theta_momentum = theta_momentum - velocity

# New data
xnew = np.array([[0], [2]])
Xnew = np.c_[np.ones((2, 1)), xnew, xnew**2]

ypredict_sgd = Xnew @ theta_sgd
ypredict_momentum = Xnew @ theta_momentum

# Calculate MSE for with and without momentum
mse_sgd = MSE(y[-2:], ypredict_sgd)
mse_momentum = MSE(y[-2:], ypredict_momentum)

#plotting
plt.plot(xnew, ypredict_sgd, "b-", label=f"SGD Without Momentum (MSE: {mse_sgd:.2f})")
plt.plot(xnew, ypredict_momentum, "g-", label=f"SGD With Momentum (MSE: {mse_momentum:.2f})")

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'SGD with Adam')
plt.legend()
plt.show()




