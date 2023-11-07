import numpy as np

def R2(y_data, y_model):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)

def MSE(y_data,y_model):
    n = np.size(y_model)
    return np.sum((y_data-y_model)**2)/n

def accuracy(y_data, y_model):
    acc = 0
    for i in range(len(y_data)):
        if y_data[i] == y_model[i]:
            acc += 1
    return acc/len(y_data)

def cross_entropy(y_data, y_model):
    cost = np.sum(y_data @ y_model - np.log(1 + np.exp(y_model)))
    return cost
