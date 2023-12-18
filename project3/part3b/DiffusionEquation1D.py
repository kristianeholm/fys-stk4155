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
t = np.linspace(0,  0.15, datapoints)


z = diffusionEquation_solution(x, t)


def plot_surface(x2, y2):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x2, y2)

    Z = diffusionEquation_solution(X, T)

    cs = ax.plot_surface(X, T, Z, label="leastsquare surface", cmap=cm.coolwarm)

    cbar = plt.colorbar(cs)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('solution u(x, t)')
    ax.set_title("1D diffusion analytic solution")
    plt.show()

plot_surface(x,t)

def forward_Euler(dx, dt):

    # the points we want to find solution for
    x = np.arange(0, 1 + dx, dx)
    t = np.arange(0, 0.15 + dt, dt)

    ##the grid  n x n
    u = np.zeros((len(t), len(x)))

    #inital conditions
    u[0, :] = np.sin(np.pi*x)
    u[0, -1] = 0


    print(len(x), len(t))

    ##applying the forward euler to find u(x, t+1),

    for j in range(0, len(t) - 1):

        for i in range(1, len(x)-1):

            uxx = (u[j, i + 1] - 2*u[j, i] + u[j, i - 1])/(dx*dx)

            u[j+1, i] = uxx*dt + u[j, i]





    """fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    T, X = np.meshgrid(x, t)
    cs = ax.plot_surface(T, X, u, label="leastsquare surface", cmap=cm.coolwarm)
    cbar = plt.colorbar(cs)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x, t)')
    ax.set_title("1D diffusion equation numeric solution")
    plt.show()"""


    return u, x, t


#calculating for different dx and dy values
for dx in [0.1, 0.01]:
    #dx = 0.01
    dy = (dx * dx) / 100
    print("dx: ", dx)
    print("dy: ", dy)
    u_numeric, x, t = forward_Euler(dx, dy)

    t_size = len(t)
    x_size = len(x)

    ##test
    ##we'll use the mse, comparing z and u_numeric

    mse_time = np.zeros(t_size)

    for j in range(t_size):
        exact = diffusionEquation_solution(x, t[j])
        numeric = u_numeric[j, :]

        mse = ((exact - numeric) ** 2).mean(axis=None)
        mse_time[j] = mse

    plt.plot(t, np.log(mse_time), label=f"dx = {dx}")
    plt.legend()
    plt.ylabel("log(Mse)")
    plt.xlabel("time")

plt.show()




"""##grid test
u_grid = np.zeros((t_size, x_size))

for j in range(t_size):
    for i in range(x_size):
        exact = diffusionEquation_solution(x[i], t[j])
        numeric = u_numeric[j, i]
        u_grid[j, i] = numeric - exact

from matplotlib import ticker, cm
##plotting the contourf
X, T = np.meshgrid(x, t)
cs = plt.contourf(T, X, u_grid,
                  locator=ticker.LogLocator(),
                  cmap="autumn")

cbar = plt.colorbar(cs)

plt.title('Absolute error between analytic and numeric solution')
plt.ylabel("t")
plt.xlabel("x")
plt.show()"""











