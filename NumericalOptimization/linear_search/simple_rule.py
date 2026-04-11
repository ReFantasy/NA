import sys
import jax
import jax.numpy as jnp


def simple_rule(objfun, xk, dk, a0=0.0, b0=sys.float_info.max, alpha0=1.0, rho=0.3, gradfun=None):
    if gradfun == None:
        gradfun = jax.grad(objfun)

    def phi(xk, dk, alpha):
        y = objfun(xk + alpha * dk)
        return y

    t = 0.9
    _, _, alphal = a0, b0, alpha0
    l = 0

    while True:
        if phi(xk, dk, alphal) <= objfun(xk) + rho * jnp.dot(gradfun(xk), dk) * alphal:
            error = objfun(xk) - objfun(xk + alphal * dk)
            return alphal, l, error
        else:
            alphal = t * alphal
            l += 1


def simple_shrink(phi: callable, alpha0=1.0, scaling=0.7):
    alphal = alpha0
    k = 0
    while phi(alphal) > phi(0):
        alphal *= scaling
        k += 1

        if k > 10000:
            # print("Warning: simple_shrink did not converge after 10000 iterations.")
            break

    return alphal, phi(alphal), k
