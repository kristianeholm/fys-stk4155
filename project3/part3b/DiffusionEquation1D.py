import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def diffusionEquation_solution(x, t):
    L = 1

    F = np.sin(np.pi/L * x)
    G = np.exp(-(np.pi/L)**2 * t)

    return F*G


datapoints = 400
x = np.linspace(0, 1, datapoints)
t = np.linspace(0,  0.1, datapoints)


z = diffusionEquation_solution(x, t)


def plot_surface(x2, y2):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x2, y2)

    Z = diffusionEquation_solution(X, T)

    ax.plot_surface(X, T, Z, label="leastsquare surface", cmap=cm.coolwarm)


    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('solution u(x, t)')
    ax.set_title("1D diffusion equation solution")
    plt.show()

plot_surface(x,t)

def forward_Euler(dx, dt):

    # the points we want to find solution for
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, 0.1 + dt, dt)

    ##the grid  n x n
    u = np.zeros((len(t), len(x)))

    #inital conditions
    u[0, :] = np.sin(np.pi*x)
    u[0, -1] = 0

    ##applying the forward euler to find u(x, t+1),

    print(len(x), len(t))
    print(u)
    for j in range(0, len(t)-1):
        for i in range(1, len(x)-1):

            uxx = (u[j, i + 1] - 2*u[j, i] + u[j, i - 1])/(dx*dx)

            u[j+1, i] = uxx*dt + u[j, i]



    print(u)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    T, X = np.meshgrid(x, t)
    ax.plot_surface(T, X, u, label="leastsquare surface", cmap=cm.coolwarm)

    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    ax.set_title("1D diffusion equation numeric solution")
    plt.show()


    return 2

dx = 0.1
forward_Euler(dx, (dx*dx)/10)



