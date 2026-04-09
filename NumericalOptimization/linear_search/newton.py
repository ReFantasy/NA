from jax import grad, hessian
import jax.numpy as jnp


## 牛顿法
def newton(phi: callable, a: float, b: float, epsilon: float, gradfun=None, hessianfun=None):
    # automatically compute the gradient and hessian of the objective function
    if gradfun == None:
        gradfun = grad(phi)
    if hessianfun == None:
        hessianfun = hessian(phi)

    xk = (a + b) / 2
    k = 0

    while True:
        k += 1
        if jnp.linalg.norm(gradfun(xk)) <= epsilon:
            xstar = xk
            fstar = phi(xk)
            return xstar, fstar, k
        else:
            xk -= gradfun(xk) / hessianfun(xk)  # 式（2-43）
