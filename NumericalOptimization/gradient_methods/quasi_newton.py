import jax
import jax.numpy as jnp
import NumericalOptimization
from NumericalOptimization.utils import proj_pd, is_pd
from NumericalOptimization.linear_search import LineSearchParams


## 拟牛顿法
def quasi_newton(objfun, x0, epsilon, gradfun=None, type: str = "BFGS", line_search_function: callable = None):
    if gradfun is None:
        gradfun = jax.grad(objfun)

    xk = x0.copy()
    Hk = jnp.eye(len(xk))
    k = 0

    # 循环迭代
    gk = gradfun(xk)  # 计算梯度
    while True:
        if jnp.linalg.norm(gk) <= epsilon or k >= 1000:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            xk_old = xk.copy()  # 上一迭代点
            dk = -jnp.dot(Hk, gk)  # 拟牛顿方向

            lambdak, _, _ = line_search_function(objfun, xk, dk)  # 一维搜索

            xk += lambdak * dk  # 迭代

            gk_old = gk.copy()  # 上一梯度
            gk = gradfun(xk)  # 计算梯度

            pk = xk_old - xk  # 位移
            qk = gk_old - gk  # 梯度差
            pk = jnp.expand_dims(pk, axis=0)
            qk = jnp.expand_dims(qk, axis=0)

            # Hk_DFP
            if type == "DFP":
                Hk += (pk.T @ pk) / (pk @ qk.T) - (Hk @ qk.T @ qk @ Hk) / (qk @ Hk @ qk.T)
            # Hk_BFGS
            elif type == "BFGS":
                Hk += ((1.0 + qk @ Hk @ qk.T) / (pk @ qk.T)) * ((pk.T @ pk) / (pk @ qk.T)) - (
                    (pk.T @ qk @ Hk + Hk @ qk.T @ pk) / (pk @ qk.T)
                )

            k += 1


## 主程序
if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    epsilon = 0.001

    @jax.jit
    def objfun(x):
        y = 4.0 * (1 - x[0]) ** 2 + 5.0 * (x[1] - x[0] ** 2) ** 2
        return y

    # 重载线搜索函数
    class LineSearchFunction(NumericalOptimization.utils.LineSearchFunction):
        def __init__(self, line_search_params: LineSearchParams = LineSearchParams()):
            self.line_search_params = line_search_params

        def __call__(self, objfun, xk, dk):
            phi = lambda alpha: objfun(xk + alpha * dk)

            return NumericalOptimization.linear_search.simple_shrink(phi, alpha0=2.0, scaling=0.7)
            # return NumericalOptimization.linear_search.simple_sampled(phi, a=0.0, b=2.0, num_samples=30000)
            # return NumericalOptimization.linear_search.golden(phi, a=0.0, b=3.0, epsilon=0.001)

    search = LineSearchFunction()

    # search = NumericalOptimization.utils.LineSearchFunction()

    ## 第1组
    x0 = jnp.array([2.0, 1.0])
    xstar, fstar, k = quasi_newton(objfun, x0, epsilon, line_search_function=search)
    print(xstar, fstar, k)

    ## 第2组
    x0 = jnp.array([-2.0, 3.0])
    xstar, fstar, k = quasi_newton(objfun, x0, epsilon, line_search_function=search)
    print(xstar, fstar, k)

    ## 第3组
    x0 = jnp.array([-3.0, 2.0])
    xstar, fstar, k = quasi_newton(objfun, x0, epsilon, line_search_function=search)
    print(xstar, fstar, k)

    # [0.99997687 0.99990834] 1.2449352847303168e-08 18
    # [1.00002714 1.0000526 ] 2.9611153115907053e-09 11
    # [1.00005207 1.00011719] 1.169643854201078e-08 35
