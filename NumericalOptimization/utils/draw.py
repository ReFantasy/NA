import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def fun(x):
#    y = 100*(x[1]-x[0]**2)**2+(1.0-x[0])**2
#    y = (6+x[0]+x[1])**2+(2-3*x[0]-3*x[1]-x[0]*x[1])**2
#    y = 20+x[0]**2+x[1]**2-10.*np.cos(2.*np.pi*x[0])-10.*np.cos(2.*np.pi*x[1])
    y = ((x[0]-3.)*(x[0]+4.))**2+((x[1]-3.)*(x[1]+4.))**2
#    y = x[0]**4+x[0]*x[1]+(1+x[1])**2

    return y


if __name__ == '__main__':
    x = np.linspace(-5,5,200)
    X1,X2 = np.meshgrid(x,x)
    Z = np.zeros_like(X1)
    for i in range(200):
        for j in range(200):
            X = np.array([X1[i,j],X2[i,j]])
            Z[i,j] = fun(X)
    
    fig = plt.figure(1)
    ax = plt.axes(projection="3d")
    ax.plot_surface(X1,X2,Z,alpha=0.9,cstride=1,rstride=1,cmap='rainbow')
    
    plt.show()