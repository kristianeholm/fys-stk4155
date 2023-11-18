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
    #print('Lengden var ' + str(len(y_data)))
    return acc/len(y_data)
