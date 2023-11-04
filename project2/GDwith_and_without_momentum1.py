#GD with and witout momentum
from numpy import asarray
from numpy import arange
from numpy.random import rand
from numpy.random import seed
from matplotlib import pyplot

# Objective function --> simple polynomial
def objective(x):
    return 4 + 3 * x + 2 * x**2

# Derivative polynomial
def derivative(x):
    return 3 + 4 * x

# Gradient descent algorithm
def gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum=None):
    solutions, scores = [], []
    solution = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
    change = 0.0

    for i in range(n_iter):
        gradient = derivative(solution)

        #with and without momentum
        if momentum is not None:
            new_change = step_size * gradient + momentum * change
            #if momentum is present add momemtum
        else:
            new_change = step_size * gradient
        

        solution = solution - new_change
        change = new_change
        solution_eval = objective(solution)

        solutions.append(solution)
        scores.append(solution_eval)

    return solutions, scores

# Set the random seed for reproducibility
seed(4)

# Define the range for input
bounds = asarray([[-1.0, 1.0]])

# Define the total iterations
n_iter = 30

# step_size = learning rate
step_size = 0.2

# Define momentum
momentum = 0.05

#GD with momentum
solutions_with_momentum, scores_with_momentum = gradient_descent(objective, derivative, bounds, n_iter, step_size, momentum)

#GD without momentum
solutions_without_momentum, scores_without_momentum = gradient_descent(objective, derivative, bounds, n_iter, step_size)

# Sample input range uniformly at 0.1 increments
inputs = arange(bounds[0, 0], bounds[0, 1] + 0.1, 0.1)

# Compute targets
results = objective(inputs)

# Create a line plot of input vs result
pyplot.plot(inputs, results, label='Objective Function')

# Plot the solutions found with momentum
pyplot.plot(solutions_with_momentum, scores_with_momentum, '.-', color='red', label='GD with Momentum')

# Plot the solutions found without momentum
pyplot.plot(solutions_without_momentum, scores_without_momentum, '.-', color='blue', label='GD without Momentum')

# Show the legend
pyplot.legend()

# Show the plot
pyplot.show()