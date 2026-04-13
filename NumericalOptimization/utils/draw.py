import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import jax
import jax.numpy as jnp

def draw1d(objfun, x_range, samples=200):
    x = jnp.linspace(x_range[0], x_range[1], samples)
    y = jax.vmap(objfun)(x)

    plt.figure(1)
    plt.plot(x, y)
    plt.show()
    
def draw2d(objfun, x_range, y_range, samples_x=200, samples_y=200, stride=1):
    x = jnp.linspace(x_range[0], x_range[1], samples_x)
    y = jnp.linspace(y_range[0], y_range[1], samples_y)
    X, Y = jnp.meshgrid(x, y)
    XY = jnp.stack([X, Y], axis=-1)
    Z = jax.vmap(jax.vmap(objfun))(XY)

    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, alpha=0.9, cstride=stride, rstride=stride, cmap="rainbow")
    plt.show()



if __name__ == "__main__":

    @jax.jit
    def fun(x):
        #    y = 100*(x[1]-x[0]**2)**2+(1.0-x[0])**2
        #    y = (6+x[0]+x[1])**2+(2-3*x[0]-3*x[1]-x[0]*x[1])**2
        #    y = 20+x[0]**2+x[1]**2-10.*np.cos(2.*np.pi*x[0])-10.*np.cos(2.*np.pi*x[1])
        y = ((x[0] - 3.0) * (x[0] + 4.0)) ** 2 + ((x[1] - 3.0) * (x[1] + 4.0)) ** 2
        #    y = x[0]**4+x[0]*x[1]+(1+x[1])**2

        return y

    # x = np.linspace(-5,5,200)
    # X1,X2 = np.meshgrid(x,x)
    # Z = np.zeros_like(X1)
    # for i in range(200):
    #     for j in range(200):
    #         X = np.array([X1[i,j],X2[i,j]])
    #         Z[i,j] = fun(X)

    # fig = plt.figure(1)
    # ax = plt.axes(projection="3d")
    # ax.plot_surface(X1,X2,Z,alpha=0.9,cstride=1,rstride=1,cmap='rainbow')

    # plt.show()

    draw2d(fun, x_range=(-5, 5), y_range=(-5, 5), samples_x=200, samples_y=200, stride=3)