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


def simple_shrink(phi: callable, alpha0=2.0, scaling=0.7):
    alphal = alpha0
    k = 0
    while phi(alphal) > phi(0):
        alphal *= scaling
        k += 1

        if k > 10000:
            # print("Warning: simple_shrink did not converge after 10000 iterations.")
            break

    return alphal, phi(alphal), k


def simple_sampled(phi: callable, a=0.0, b=3.0, num_samples=1000):
    """
    使用均匀采样法在一维区间内寻找函数的近似极小值。

    参数:
        phi (callable): 接受单个标量输入并返回其函数值的一维目标函数。
        a (float, optional): 采样区间的起点。默认值为 0.0。
        b (float, optional): 采样区间的终点。默认值为 3.0。
        num_samples (int, optional): 采样点的总数。默认值为 1000。

    返回:
        tuple: 包含三个结果的元组
            - float: 使得函数值最小的 alpha 值。
            - float: 找到的最小函数值。
            - int: 使用的采样点总数。
    """

    while phi(b) < phi(0):
        b *= 1.1

    alphas = jnp.linspace(a, b, num_samples)
    phis = jax.vmap(phi)(alphas)
    min_idx = jnp.argmin(phis)
    return alphas[min_idx], phis[min_idx], num_samples
