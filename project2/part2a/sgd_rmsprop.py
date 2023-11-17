#SGD with and without momentum + rmsprop
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


n = 400 #darapoints
x = 2 * np.random.rand(n, 1)
y = 4 + 3 * x + 2 * x**2 + np.random.randn(n, 1)  # f(x) = a_0 + a_1x + a_2x^2

X = np.c_[np.ones((n, 1)), x, x**2] #design matrix

theta_linreg = np.linalg.inv(X.T @ X) @ X.T @ y


# Note that we request the derivative wrt third argument (theta, 2 here)
training_gradient = grad(CostOLS, 2)

# number of epochs and minibatches
n_epochs = 100
minibatch_size = 25


t0, t1 = 5, 40  # learning schedule parameters

# learning schedule function
def learning_schedule(t):
    return t0 / (t + t1)


# Initialization of Theta for SGD without momentum
theta_sgd = np.random.randn(3, 1)
#theta for with momentum
theta_momentum = np.random.randn(3, 1)

# RMSprop parameters
rho = 0.99
delta = 1e-8  # adding the adagrad parameter to avoid division by zero

G = np.zeros(theta_sgd.shape) #no momentum
G_momentum = np.zeros(theta_momentum.shape) #momentum

# SGD without momentum + RMSprop
for epoch in range(n_epochs):
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index + minibatch_size]
        yi = y[random_index:random_index + minibatch_size]

        gradients_no_momentum = (1.0 / minibatch_size) * training_gradient(yi, xi, theta_sgd)

        eta = learning_schedule(epoch * n // minibatch_size + i) #from learning schedule

        # RMSprop update without momentum
        G = (rho * G + (1 - rho) * gradients_no_momentum**2)

        # Update parameters with RMSprop (without momentum)
        update_no_momentum = gradients_no_momentum * eta / (np.sqrt(G) + delta)
        theta_sgd -= update_no_momentum

#velocity for SGD with momentum
velocity = np.zeros((3, 1))
mu = 0.3  # momentum

# SGD with momentum + RMSprop
for epoch in range(n_epochs):
    for i in range(n // minibatch_size):
        random_index = np.random.randint(n)
        xi = X[random_index:random_index + minibatch_size]
        yi = y[random_index:random_index + minibatch_size]

        # Calculate gradients
        gradients_momentum = (1.0 / minibatch_size) * training_gradient(yi, xi, theta_momentum)

        # RMSprop update with momentum
        G_momentum = (rho * G_momentum + (1 - rho) * gradients_momentum**2)

        eta = learning_schedule(epoch * n // minibatch_size + i) #from learning schedule

        # Update parameters with RMSprop (with momentum)
        velocity = mu * velocity + eta * gradients_momentum
        theta_momentum = theta_momentum - velocity

# New data
xnew = np.array([[0], [2]])
Xnew = np.c_[np.ones((2, 1)), xnew, xnew**2]

#predicted data
ypredict_sgd = Xnew @ theta_sgd
ypredict_momentum = Xnew @ theta_momentum

# Calculate MSE for with and without momentum
mse_sgd = MSE(y[-2:], ypredict_sgd)
mse_momentum = MSE(y[-2:], ypredict_momentum)

#PLOTTING
plt.plot(xnew, ypredict_sgd, "b-", label=f"SGD Without Momentum (MSE: {mse_sgd:.2f})")
plt.plot(xnew, ypredict_momentum, "g-", label=f"SGD With Momentum (MSE: {mse_momentum:.2f})")

plt.xlabel(r'$x$')
plt.ylabel(r'$y$')
plt.title(r'SGD with RMSprop')
plt.legend()
plt.show()