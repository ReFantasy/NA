import jax
import jax.numpy as jnp
from NumericalOptimization.utils import linprog, linear_search
from loguru import logger


def frank_wolfe(objfun, A, b, E, e, x0, gradfun=None, max_iter=1000, atol: float = 1e-4, verbose=True):
    jnp.set_printoptions(formatter={"float": "{: .4e}".format})

    if gradfun == None:
        gradfun = jax.grad(objfun)

    k = 0
    xk = x0

    while True:
        grad = gradfun(xk)
        y, _ = linprog(f=grad, A=A, b=b, Aeq=E, beq=e, lb=-jnp.inf, ub=jnp.inf)

        d = y - xk
        if jnp.isclose(jnp.dot(grad, d), 0.0, atol=atol):
            if verbose:
                logger.info(f"Iter: {k}, xk: {xk}, objfun(xk): {objfun(xk)}, grad @ d: {jnp.dot(grad, d)}")
            return xk, objfun(xk)
        else:
            phi = lambda lam: objfun(xk + lam * d)
            lam, _, _ = linear_search.golden(phi, a=0.0, b=1.0, epsilon=atol)

        xk = xk + lam * d

        k = k + 1
        if verbose:
            logger.info(f"Iter: {k}, xk: {xk}, objfun(xk): {objfun(xk)}, grad @ d: {jnp.dot(grad, d)}")
        if k >= max_iter:
            logger.warning("Maximum iterations reached without convergence.")
            return xk, objfun(xk)


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def objfun(x):
        x1, x2, x3, x4 = x
        return x1**2 + x2**2 - x1 * x2 - 2 * x1 + 3 * x2

    A = jnp.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=jnp.float64)
    b = jnp.array([0.0, 0.0, 0.0, 0.0])
    E = jnp.array([[1, 1, 1, 0], [1, 5, 0, 1]], dtype=jnp.float64)  # 没有等式约束
    e = jnp.array([3.0, 6.0], dtype=jnp.float64)  # 没有等式约束的右侧值

    x0 = jnp.array([2.6259, 0.2312, 0.1429, 2.2179])
    xstar, fstar = frank_wolfe(objfun, A, b, E, e, x0=x0, verbose=True, atol=0.01)
    print(f"Optimal solution: {xstar}, optimal value: {fstar}")

    # from NumericalOptimization.constrained_optimization import zoutendijk_lin
    # xstar, fstar, _ = zoutendijk_lin(objfun, A, b, E, x0=x0, verbose=True,atol=0.001)
    # print(f"Optimal solution: {xstar}, optimal value: {fstar}")
