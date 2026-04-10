from NumericalOptimization import utils
import jax
import jax.numpy as jnp
from NumericalOptimization.linear_search import LineSearchParams


## 最速下降法
def gradient_descent(objfun, x0, epsilon, gradfun=None, line_search_params: LineSearchParams = LineSearchParams()):
    if gradfun is None:
        gradfun = jax.grad(objfun)

    xk = x0
    k = 0
    while True:
        k += 1
        gk = gradfun(xk)
        if jnp.linalg.norm(gk) <= epsilon:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            dk = -gk

            # linear search
            alpha, _, _ = utils.line_search_function(objfun, xk, dk, line_search_params)

            xk += alpha * dk
