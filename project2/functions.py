import numpy as np

def sigmoid(z):
    z = np.clip(z, -700, 700)
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return np.where(z > 0, 1, 0)

def relu_leaky(z):
    return np.where(z > 0, z, 0.01*z)

def relu_leaky_derivative(z):
    return np.where(z > 0, 1, 0.01)