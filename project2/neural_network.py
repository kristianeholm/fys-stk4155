import numpy as np
from sklearn.utils import shuffle

from functions import sigmoid, sigmoid_derivative, relu, relu_derivative, relu_leaky, relu_leaky_derivative
from metrics import MSE, R2

class NeuralNetwork:
#Credits to https://github.com/CompPhysics/MachineLearning/blob/master/doc/LectureNotes/week41.ipynb from which this class reuses some code.
    def __init__(
            self,
            num_features, 
            regr_or_class='regr', 
            activation = 'sigmoid',
            #X_data,
            #Y_data,
            #n_hidden_neurons=50,
            #n_categories=10,
            epochs=10,
            minibatches=5,
            eta=0.1,
            lmbd=0.0):

#        self.X_data_full = X_data
#        self.Y_data_full = Y_data

#        self.n_inputs = X_data.shape[0]
#        self.n_features = X_data.shape[1]
#        self.n_hidden_neurons = n_hidden_neurons
#        self.n_categories = n_categories

        self.epochs = epochs
        self.minibatches = minibatches
#        self.iterations = self.n_inputs // self.batch_size
        self.learning_rate = eta
        self.lmbd = lmbd
        
        self.num_features = num_features
        self.weights = {}
        self.biases = {}
        self.activations = {}
        self.errors = {}
        self.learning_type = regr_or_class
        if activation == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = relu
            self.activation_prime = relu_derivative
        elif activation == 'leakyrelu':
            self.activation_function = relu_leaky
            self.activation_prime = relu_leaky_derivative

    def initialize_weights(self, size_layer, size_prev_layer):
        return np.random.randn(size_prev_layer, size_layer) 
    
    def initialize_bias(self, size_layer):
        return np.random.rand(size_layer)
        
    def add_layer(self, size_layer):
        if len(self.weights) == 0:
            self.weights[0] = self.initialize_weights(size_layer, self.num_features)
            self.biases[0] = self.initialize_bias(size_layer)
        else:
            counter = len(self.weights)
            size_prev_layer = self.weights[counter - 1].shape[1]
            self.weights[counter] = self.initialize_weights(size_layer, size_prev_layer)
            self.biases[counter] = self.initialize_bias(size_layer)
            
    def compute_z(self, current_layer):
        weights = self.weights.get(current_layer)
        bias = self.biases.get(current_layer)
        inputs = self.activations.get(current_layer)
        z = inputs @ weights + bias
        return z
            
    def feed_forward(self):
        current_layer = 0
        num_layers = len(self.weights)
        while current_layer < num_layers:
            z = self.compute_z(current_layer)
            if current_layer == num_layers - 1:
                if self.learning_type == 'class':
                    a = sigmoid(z)
                else:
                    a = z
            else:
                a = self.activation_function(z)
            current_layer += 1
            self.activations[current_layer] = a

    def backpropagation(self, y):
        current_layer = len(self.weights)
        a = self.activations.get(current_layer).ravel()
        C_deriv = (a - y).reshape(-1, 1)
        z = self.compute_z(current_layer-1)
        if self.learning_type == 'regr':
            activation_deriv = np.ones((len(z), 1))
        elif self.learning_type == 'class':
            activation_deriv = sigmoid_derivative(z)
        output_error = C_deriv * activation_deriv
        self.errors[current_layer] = output_error
        current_layer -= 1
        while current_layer > 0:
            error_prev = self.errors[current_layer + 1]
            weights = self.weights[current_layer]
            z = self.compute_z(current_layer-1)
            activation_deriv = self.activation_prime(z)
            error = np.dot(error_prev, weights.T) * activation_deriv
            self.errors[current_layer] = error
            current_layer -= 1
            
    def update_weights(self):
        current_layer = 0
        while current_layer < len(self.weights):
            activations = self.activations[current_layer]
            error = self.errors[current_layer + 1]
            self.weights[current_layer] = (1-self.learning_rate*self.lmbd/len(activations))*self.weights[current_layer] - self.learning_rate*np.dot(activations.T, error)
            self.biases[current_layer] = self.biases[current_layer] - self.learning_rate*np.sum(error, axis=0)
            current_layer += 1
        
    def train(self, data, target, data_val=None, target_val=None): 
        minibatches = 1
        n = len(data)
        batch_size = int(n/minibatches)
        
        self.loss_train = []
        self.loss_val = []
        
        for i in range(self.epochs):
            data_shuffle, target_shuffle = shuffle(data, target)
            batch_chosen = np.random.randint(0, minibatches)
            data_minibatch = data_shuffle[batch_chosen:batch_chosen+batch_size]
            target_minibatch = target_shuffle[batch_chosen:batch_chosen+batch_size]
            self.activations[0] = data_minibatch
            self.feed_forward()
            self.backpropagation(target_minibatch)
            self.update_weights()
            
            target_pred_val = self.predict(data_val)
            val_loss = MSE(target_pred_val, target_val)
                
            target_pred_train = self.predict(data)
            train_loss = MSE(target_pred_train, target)
            
            self.loss_val.append(val_loss)
            self.loss_train.append(train_loss)

    def predict(self, x):
        self.activations[0] = x
        self.feed_forward()
        preds = self.activations[len(self.activations) - 1].ravel()
        if self.learning_type == 'class':
            return np.where(preds >= 0.5, 1, 0)
        return preds
        
    def predict_probabilities(self, x):
        self.feed_forward(x)
        preds = self.activations[len(self.activations) - 1]
        return preds