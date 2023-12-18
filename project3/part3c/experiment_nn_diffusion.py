import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_network_pde_diffusion import NeuralNetworkPDEDiffusion
import matplotlib.pyplot as plt
from matplotlib import cm

#Hyper-parameters selected
learning_rate=0.01
activation_function="relu"
num_layers=2
nodes_per_layer = 200
epochs = 300

#Defining the amount of data
datapointsx = 400
datapointst = 50

#Random seed fixed for more stable results while experimenting with different NN
seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

#The analytical solution, for comparing against for exact error. 
def diffusionEquation_solution(x, t):
    L = 1

    F = np.sin(np.pi/L * x)
    G = np.exp(-(np.pi/L)**2 * t)

    return F*G

x = np.linspace(0, 1, datapointsx)
t = np.linspace(0,  0.15, datapointst)
    
x, t = np.meshgrid(x, t)
x, t = x.ravel(), t.ravel()

#Fit the model
layers = [nodes_per_layer]*num_layers + [1]
my_model = NeuralNetworkPDEDiffusion(layers=layers, activation_function=activation_function, learning_rate=learning_rate)
print("\n")
loss = my_model.train_model(x=x, t=t, epochs=epochs)
epochs_array = np.linspace(1, epochs, epochs)

print("Final MSE value ", loss[-1])
#Plot loss vs epochs
plt.plot(epochs_array, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.savefig("neural_network_loss_function_{}_{}_{}.pdf".format(num_layers, datapointsx, learning_rate))
plt.show()


# Define grid and compute predicted solution
num_points = 400
start = tf.constant(0.01, dtype=tf.float32)
stop = tf.constant(0.99, dtype=tf.float32)
start_t = tf.constant(0, dtype=tf.float32)
stop_t = tf.constant(0.15, dtype=tf.float32)
X, T = tf.meshgrid(tf.linspace(start, stop, num_points), tf.linspace(start_t, stop_t, num_points))
x, t = tf.reshape(X, [-1, 1]), tf.reshape(T, [-1, 1])
f_predict = my_model.predict(x, t)
g = tf.reshape(f_predict, (num_points, num_points))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, g, label="leastsquare surface", cmap=cm.coolwarm)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('NN solution u(x, t)')
ax.set_title("Neural Network 1D diffusion equation solution")

plt.savefig("neural_network_solution_{}_{}_{}.pdf".format(num_layers, datapointsx, learning_rate))
#plt.savefig('neural_network_solution_.pdf')
plt.show()

# Also compute exact solution and error for these points.
g_exact = tf.reshape(diffusionEquation_solution(x, t), (num_points, num_points))
rel_err = np.abs((g - g_exact)/g)
abs_err = np.abs((g - g_exact))

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, g_exact, label="leastsquare surface", cmap=cm.coolwarm)

plt.title('Loss function as MSE over diff eq equiality per epoch')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Exact solution u(x, t)')
ax.set_title("Exact 1D diffusion equation solution")
plt.savefig('exact_solution_diffusion_.pdf')
plt.show()

#Plot absolute and relative errors.

from matplotlib import ticker, cm
##plotting the contourf
#X, T = np.meshgrid(x, t)
cs = plt.contourf(X, T, abs_err,
                  locator=ticker.LogLocator(),
                  cmap="autumn")

cbar = plt.colorbar(cs)

plt.title('Absolute error between analytic and Neural Network')
plt.ylabel("t")
plt.xlabel("x")
plt.savefig('absolute_error_NN_diffusion_.pdf')
plt.show()

cs = plt.contourf(X, T, rel_err,
                  locator=ticker.LogLocator(),
                  cmap="autumn")

cbar = plt.colorbar(cs)

plt.title('Relative error between analytic and Neural Network')
plt.ylabel("t")
plt.xlabel("x")
plt.savefig('relative_error_NN_diffusion_.pdf')
plt.show()


#Now also plot MSE to get a fair comparasion with Forward Euler
#t_size = len(t)
#print("Length t ", len(t))
#
##we'll use the mse, comparing z and u_numeric
#
#mse_time = np.zeros(t_size)
#
#for j in range(t_size):
#    #print('Getting MSE for t=', j)
#    exact = diffusionEquation_solution(x, t[j])
#    #f_predict = my_model.predict(x, t)
#    #g = tf.reshape(f_predict, (num_points, num_points))
#    numeric = f_predict[j, :]
#
#    mse = ((exact - numeric) ** 2).numpy().mean(axis=None)
#    mse_time[j] = mse
#
#plt.plot(t, np.log(mse_time), label=f"dx")
#plt.legend()
#plt.ylabel("log(Mse)")
#plt.xlabel("time")
#
#plt.savefig('log_mse.pdf')
#plt.show()