import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from neural_network_pde_diffusion import NeuralNetworkPDEDiffusion
import matplotlib.pyplot as plt
from matplotlib import cm



seed = 1234
tf.random.set_seed(seed)
np.random.seed(seed)

#The analytical solution, for comparing against for exact error. 
def diffusionEquation_solution(x, t):
    L = 1

    F = np.sin(np.pi/L * x)
    G = np.exp(-(np.pi/L)**2 * t)

    return F*G

datapointsx = 200
datapointst = 50
x = np.linspace(0, 1, datapointsx)
t = np.linspace(0,  0.15, datapointst)
    
x, t = np.meshgrid(x, t)
x, t = x.ravel(), t.ravel()

print(x.size)
print(t.size)

#Fit the model
layers = [100]*4 + [1]
input_sz = 2
epochs = 300
my_model = NeuralNetworkPDEDiffusion(layers=layers, input_sz=input_sz, learning_rate=0.01)
print("\n")
loss = my_model.train_model(x=x, t=t, epochs=epochs)
epochs_array = np.linspace(1, epochs, epochs)

print("Final MSE value ", loss[-1])
#Plot loss vs epochs
plt.plot(epochs_array, loss)
plt.xlabel("Epochs")
plt.ylabel("Loss")
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
plt.show()

# Also compute exact solution and error for these points.
g_exact = tf.reshape(diffusionEquation_solution(x, t), (num_points, num_points))
rel_err = np.abs((g - g_exact)/g)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, g_exact, label="leastsquare surface", cmap=cm.coolwarm)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('Exact solution u(x, t)')
ax.set_title("Exact 1D diffusion equation solution")
plt.show()

#Plot relative error
fig = plt.figure()
ax = fig.add_subplot(111)
fontsize = 16
ticksize = 16
plt.pcolormesh(X, T, rel_err, cmap="inferno")
cbar = plt.colorbar()
cbar.set_label("Relative error", size=fontsize)
cbar.ax.tick_params(labelsize=ticksize)
cbar.formatter.set_powerlimits((0,0))
cbar.update_ticks()
plt.xticks(size=ticksize)
plt.yticks(size=ticksize)
ax.set_xlabel(r"$x$", size=fontsize)
ax.set_ylabel(r"$t$", size=fontsize)

