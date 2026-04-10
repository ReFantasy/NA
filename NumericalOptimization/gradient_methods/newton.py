import jax
import jax.numpy as jnp
from NumericalOptimization.utils import line_search_function, proj_pd, is_pd
from NumericalOptimization.linear_search import LineSearchParams


def newton_basic(objfun, x0, epsilon, gradfun=None, hessianfun=None, line_search_params: LineSearchParams = None):
    if gradfun is None:
        gradfun = jax.grad(objfun)
    if hessianfun is None:
        hessianfun = jax.hessian(objfun)

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
            Gk = hessianfun(xk)

            # Gk_inv = jnp.linalg.inv(Gk)
            # dk = -jnp.dot(Gk_inv, gk)

            # here we can use jnp.linalg.solve to solve the linear system Gk * dk = -gk,
            # which is more efficient and numerically stable than computing the inverse of Gk
            dk = jnp.linalg.solve(Gk, -gk)

            # if line_search_name is not None, we can perform line search to find the optimal step size alpha
            # this is called damped Newton method, which can improve the convergence of the algorithm
            if line_search_params is not None:
                alpha, _, _ = line_search_function(objfun, xk, dk, line_search_params)
                xk += alpha * dk
            else:
                xk += dk


def newton_goldstein(objfun, x0, epsilon, gradfun=None, hessianfun=None, line_search_params: LineSearchParams = None):
    """
    if hessian of objective function is not positive definite, the Newton direction may not be a descent direction,
    so we can use the gradient direction to replace the Newton direction for line search, which is called Goldstein-Price method.
    """
    if gradfun is None:
        gradfun = jax.grad(objfun)
    if hessianfun is None:
        hessianfun = jax.hessian(objfun)

    eta = 0.3
    xk = x0.copy()
    k = 0
    while True:
        k += 1
        gk = gradfun(xk)
        if jnp.linalg.norm(gk) <= epsilon:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            Gk = hessianfun(xk)  # 海森矩阵
            # Gk_inv = jnp.linalg.inv(Gk)      # 海森矩阵的逆
            # dkN = -jnp.dot(Gk_inv,gk)        # 牛顿方向
            dkN = jnp.linalg.solve(Gk, -gk)

            # 当Gk非正定时，牛顿方向可能不是下降方向，此时可以使用最速下降方向进行修正
            coss = jnp.dot(dkN, -gk) / (jnp.linalg.norm(dkN) * jnp.linalg.norm(-gk))
            if coss >= eta:
                dk = dkN  # 使用牛顿方向搜索
            else:
                dk = -gk  # 使用最速下降方向搜索

            if line_search_params is not None:
                alpha, _, _ = line_search_function(objfun, xk, dk, line_search_params)
                xk += alpha * dk
            else:
                xk += dk


def newton_goldfeld(objfun, x0, epsilon, gradfun=None, hessianfun=None, line_search_params: LineSearchParams = None):
    """
    if hessian of objective function is not positive definite, the Newton direction may not be a descent direction,
    so we can modify the Hessian matrix by adding a positive multiple of the identity matrix to make it positive definite, which is called Goldfeld method.
    """
    if gradfun is None:
        gradfun = jax.grad(objfun)
    if hessianfun is None:
        hessianfun = jax.hessian(objfun)

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
            Gk = hessianfun(xk)

            # 当Gk非正定时，牛顿方向可能不是下降方向，因此需要对海森矩阵进行修正
            # 常用的方法是加上一个正数乘以单位矩阵，使得修正后的矩阵变为正定矩阵
            if not is_pd(Gk):
                Gk = proj_pd(Gk)

            # Gk_inv = jnp.linalg.inv(Gk)
            # dk = -jnp.dot(Gk_inv, gk)

            # here we can use jnp.linalg.solve to solve the linear system Gk * dk = -gk,
            # which is more efficient and numerically stable than computing the inverse of Gk
            dk = jnp.linalg.solve(Gk, -gk)

            # if line_search_params is not None, we can perform line search to find the optimal step size alpha
            # this is called damped Newton method, which can improve the convergence of the algorithm
            if line_search_params is not None:
                alpha, _, _ = line_search_function(objfun, xk, dk, line_search_params)
                xk += alpha * dk
            else:
                xk += dk


## 主程序
if __name__ == "__main__":

    @jax.jit
    def objfun(x):
        y = jnp.pow(x[0], 4) + x[0] * x[1] + (1.0 + x[1]) ** 2
        return y

    epsilon = 0.001
    # x0 = jnp.array([1.0, 2.0])
    x0 = jnp.array([0.0, 0.0])
    # xstar, fstar, k = newton_basic(objfun, x0, epsilon)
    # print(xstar, fstar, k)
    xstar, fstar, k = newton_goldstein(objfun, x0, epsilon, line_search_params=LineSearchParams(name="golden"))
    print(xstar, fstar, k)
    xstar, fstar, k = newton_goldfeld(objfun, x0, epsilon, line_search_params=LineSearchParams(name="golden"))
    print(xstar, fstar, k)
