import numpy as np
from sklearn.utils import shuffle

from functions import sigmoid, sigmoid_derivative, relu, relu_derivative, relu_leaky, relu_leaky_derivative
from metrics import MSE, R2, accuracy, cross_entropy

class NeuralNetwork:
#Credits to https://github.com/CompPhysics/MachineLearning/blob/master/doc/LectureNotes/week41.ipynb from which this class reuses some code.
    def __init__(
            self,
            num_features, 
            learning_type='regr', 
            activation = 'sigmoid',
            cost_function='MSE',
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
        self.learning_type = learning_type
        self.cost_function = cost_function
        if activation == 'sigmoid':
            self.activation_function = sigmoid
            self.activation_prime = sigmoid_derivative
        elif activation == 'relu':
            self.activation_function = relu
            self.activation_prime = relu_derivative
        elif activation == 'leakyrelu':
            self.activation_function = relu_leaky
            self.activation_prime = relu_leaky_derivative

    def add_layer(self, size_layer):
        number_layers_so_far = len(self.weights)
        if number_layers_so_far == 0:
            number_of_inputs = self.num_features
        else: 
            number_of_inputs = self.weights[number_layers_so_far - 1].shape[1]
        self.weights[number_layers_so_far] = np.random.randn(number_of_inputs, size_layer)
        self.biases[number_layers_so_far] = np.random.rand(size_layer)
            
    def compute_z(self, current_layer):
        weights = self.weights.get(current_layer)
        bias = self.biases.get(current_layer)
        inputs = self.activations.get(current_layer)
        return inputs @ weights + bias
            
    def feed_forward(self):
        number_of_layers = len(self.weights)
        
        #Loop over hidden layers:
        for current_layer in range(number_of_layers-1):
            z_h = self.compute_z(current_layer)
            a_h = self.activation_function(z_h)
            self.activations[current_layer+1] = a_h
            
        #Output layer:
        z_o = self.compute_z(number_of_layers-1)
        if self.learning_type == 'regr':
            self.activations[number_of_layers] = z_o
        else:
            self.activations[number_of_layers] = sigmoid(z_o)
            
    #TODO, put feed_forward_out back

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
            
    def compute_cost_function(self, prediction, target):
        if self.cost_function == 'MSE':
            return MSE(prediction, target)
        elif self.cost_function == 'R2':
            return R2(prediction, target)
        elif self.cost_function == 'accuracy':
            return accuracy(prediction, target)
        elif self.cost_function == 'logistic':
            return cross_entropy(prediction, target)
        else:
            raise Exception('Undefined cost function', self.cost_function)
    
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
        
        self.cost_train = []
        self.cost_test = []
        
        for i in range(self.epochs):
            data_shuffle, target_shuffle = shuffle(data, target)
            batch_chosen = np.random.randint(0, minibatches)
            data_minibatch = data_shuffle[batch_chosen:batch_chosen+batch_size]
            target_minibatch = target_shuffle[batch_chosen:batch_chosen+batch_size]
            self.activations[0] = data_minibatch
            self.feed_forward()
            self.backpropagation(target_minibatch)
            self.update_weights()

            test_cost = self.compute_cost_function(self.predict(data_val), target_val)
            train_cost = self.compute_cost_function(self.predict(data), target)                                                         
            
            self.cost_test.append(test_cost)
            self.cost_train.append(train_cost)

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