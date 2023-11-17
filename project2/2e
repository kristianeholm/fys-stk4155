#logistic regression (without scikit learn)
import autograd.numpy as np
from autograd import grad
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from metrics import accuracy

np.random.seed(0)

#Cross- entropy cost function
def CostCrossEntropy(target):
    def func(X):
        return -(1.0 / target.size) * np.sum(target * np.log(X + 10e-10))

    return func

#Function for making the design matrix
def create_design_matrix(x, polynomial_degree=1):
    X = pd.DataFrame()
    for i in range(1, polynomial_degree + 1):
        col_name = 'x^' + str(i)
        #X[col_name] = x**i
        X[col_name] = x[:, 0]**i  # Access the first column of x
    return X

#activation function
def sigmoid(x):
    return 0.5 * (np.tanh(x / 2.) + 1)

def logistic_predictions(weights, inputs):
    # Outputs probability of a label being true according to logistic model.
    return sigmoid(np.dot(inputs, weights))

def training_loss(weights):
    # Training loss is the negative log-likelihood of the training labels.
    preds = logistic_predictions(weights, inputs)
    label_probabilities = preds * targets + (1 - preds) * (1 - targets)
    return -np.sum(np.log(label_probabilities))

#stohastic gradient decent (from sgd and ridge from earlier)
#n_epochs = number of epochs, lmbda = lambda = regularization parameter, eta = learning rate (step size)
def sgd(X, y, n_epochs, minibatch_size, lmbda, eta):
    n, p = X.shape
    theta_sgd = np.random.randn(p, 1)

    for epoch in range(n_epochs):
        for i in range(n // minibatch_size):
            random_index = np.random.randint(n)
            xi = X[random_index:random_index + minibatch_size]
            yi = y[random_index:random_index + minibatch_size]
            #if we would want to use the learning schedule function from earlier
            #t = epoch * (n // minibatch_size) + i  # Total iteration count
            #learning_rate = eta / (1 + lmbda * t)  # Add a schedule for learning rate
            
            # Compute the gradient with the regularization term lambda
            gradients = np.dot(xi.T, (logistic_predictions(theta_sgd, xi) - yi))
            gradients += 2 * lmbda * theta_sgd
            #theta_sgd = theta_sgd - learning_rate * gradients
            theta_sgd = theta_sgd - eta * gradients

             # Print gradients and weights
            #print(f"Gradients: {gradients.flatten()}")
            #print(f"Weights: {theta_sgd.flatten()}")

    return theta_sgd


#get the breast cancer dataset
cancer=load_breast_cancer()      #Download breast cancer dataset

x = cancer.data                     #Feature matrix of 569 rows (samples) and 30 columns (parameters)
y = (cancer.target == 1).astype(int)                 #Label array of 569 rows (0 for benign and 1 for malignant)

#design matrix
X = create_design_matrix(x)

# Split the data into training and test --> training 80%, test 20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling the data with standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# use the split and scaled data into inputs and targets
inputs = X_train_scaled
targets = y_train.reshape(-1, 1)

# Train logistic regression with SGD and ridge regularization
eta = 0.01 # Set your desired fixed learning rate
lmbda = 0.001
n_epochs = 1000
minibatch_size = 50
trained_weights = sgd(inputs, targets, n_epochs=n_epochs, minibatch_size= minibatch_size, eta=eta, lmbda=lmbda) #sgd
#print("Trained weights:", trained_weights.flatten())

#See how good the model work on the test set
test_inputs = X_test_scaled
test_targets = y_test.reshape(-1, 1)
test_predictions = logistic_predictions(trained_weights, test_inputs)

"""
# Print predictions and targets for debugging
print("Test Predictions:", test_predictions.flatten())
print("Test Targets:", test_targets.flatten())
"""

accuracy = np.mean((test_predictions >= 0.5) == test_targets)

print("Test Accuracy:", accuracy)


#plotting the accuracy with respect to lambda

lmbda_values = [0.0000000001, 0.001, 0.01, 0.1, 1.0, 10.0, 100, 1000] #lambda values we want to try
accuracies = [] #where to store the accuracy for each lamda value

for lmbda in lmbda_values: 
    # Train logistic regression with SGD and ridge regularization
    trained_weights = sgd(inputs, targets, n_epochs=n_epochs, minibatch_size= minibatch_size, eta=eta, lmbda=lmbda)

    # Evaluate on the test set
    test_predictions = logistic_predictions(trained_weights, test_inputs)
    accuracy = np.mean((test_predictions >= 0.5) == test_targets)
    
    accuracies.append(accuracy)

# Plot the results
plt.plot(lmbda_values, accuracies, marker='o')
plt.xscale('log')  # Use a logarithmic scale for lambda
plt.xlabel('Regularization Parameter (lambda)')
plt.ylabel('Test Accuracy')
plt.title('Accuracy vs Regularization Parameter')
plt.show()


"""
# Plotting the training loss over number of epochs
theta_initial = np.random.randn(X_train_scaled.shape[1], 1)
training_gradient_fun = grad(training_loss)
theta_sgd = theta_initial.copy()
num_epochs = 20 #number of epochs
losses = []

for epoch in range(num_epochs):
    gradients = training_gradient_fun(theta_sgd)
    theta_sgd -= eta * gradients
    current_loss = training_loss(theta_sgd)
    losses.append(current_loss)

# Plot the training loss
plt.plot(range(1, len(losses) + 1), losses)
plt.xlabel('Epochs')
plt.ylabel('Training Loss')
plt.title('Training Loss over Epochs')
plt.show()
"""