import NumericalOptimization.linear_search as linear_search
import jax.numpy as jnp
import jax
import NumericalAnalysis as na
from NumericalOptimization.linear_search import LineSearchParams


def chase(phi: callable, x_init: float, h: float):
    """
    使用进退法（Bounding Phase Method）寻找一元函数的极小值包含区间。
    作者: LONG QIANG (Created on Thu Mar 24 19:05:38 2022)
    通过不断步进试探，找到一个呈现“高-低-高”形态的区间，该区间必定包含极小值点。

    参数
    ----------
    func : callable
        目标函数。
    x_init : float
        初始搜索点。
    h : float
        初始搜索步长。

    返回
    -------
    tuple[float, float, int]
        包含极小值点的区间端点 x1, x3，以及迭代次数 k。
    """
    x1, x2 = x_init, x_init + h
    fx1, fx2 = phi(x1), phi(x2)
    k = 0

    if fx1 > fx2:
        while True:
            k += 1
            x3 = x2 + h
            fx3 = phi(x3)
            if fx2 > fx3:
                x1, x2 = x2, x3
                fx1, fx2 = fx2, fx3
            else:
                return x1, x3, k
    else:
        x3, x2 = x2, x1
        fx3, fx2 = fx2, fx1
        while True:
            k += 1
            x1 = x2 - h
            fx1 = phi(x1)
            if fx2 < fx1:
                return x1, x3, k
            else:
                x3, x2 = x2, x1
                fx3, fx2 = fx2, fx1


def is_pd(A: jnp.ndarray) -> jnp.ndarray:
    """
    检查矩阵 A 是否对角占优。
    当A是对角占优的，则A是正定矩阵。
    当A不是对角占优也可能是正定的。
    参数
    ----------
    A : jnp.ndarray
        待检查的矩阵。

    返回
    -------
    jnp.ndarray
        如果 A 是正定矩阵，返回 True；否则返回 False。
    """
    return na.utils.is_sdd(A)


def proj_pd(A: jnp.ndarray, delta: float = 1e-2) -> jnp.ndarray:
    """
    将矩阵 A 投影到正定矩阵空间。
    参数
    ----------
    A : jnp.ndarray
        待投影的矩阵。

    返回
    -------
    jnp.ndarray
        投影后的正定矩阵。
    """
    assert A.shape[0] == A.shape[1], "Input must be a square matrix."

    while not is_pd(A):
        A += jnp.eye(A.shape[0]) * delta
    return A


def line_search_method(name="golden"):
    if name == "golden":
        return linear_search.golden
    elif name == "newton":
        return linear_search.newton
    elif name == "fibonacci":
        return linear_search.fibonacci
    elif name == "armijo_goldstein":
        return linear_search.armijo_goldstein
    elif name == "wolf_powell":
        return linear_search.wolf_powell
    elif name == "secant":
        return linear_search.secant
    elif name == "parabola":
        return linear_search.parabola
    elif name == "simple_rule":
        return linear_search.simple_rule
    else:
        raise ValueError("Unknown line search method: {}".format(name))


class LineSearchFunction:
    """
    线搜索函数类，用于在优化算法中执行线搜索。
    重载 __call__ 方法，自定义线搜索行为。
    """

    def __init__(self, line_search_params: LineSearchParams = LineSearchParams()):
        self.line_search_params = line_search_params

    def __call__(self, objfun, xk, dk):
        phi = lambda alpha: objfun(xk + alpha * dk)

        return linear_search.simple_shrink(phi)

        # line_search_params = self.line_search_params
        # method_name = line_search_params.name
        # search = line_search_method(method_name)
        # if method_name == "armijo_goldstein":
        #     lambdak, fstar, k = search(
        #         objfun=objfun,
        #         xk=xk,
        #         dk=dk,
        #         a0=line_search_params.a,
        #         b0=line_search_params.b,
        #         alpha0=line_search_params.alpha,
        #         rho=line_search_params.rho,
        #         t=line_search_params.t,
        #     )
        # elif method_name == "wolf_powell":
        #     lambdak, fstar, k = search(
        #         objfun=objfun,
        #         xk=xk,
        #         dk=dk,
        #         a0=line_search_params.a,
        #         b0=line_search_params.b,
        #         alpha0=line_search_params.alpha,
        #         rho=line_search_params.rho,
        #         t=line_search_params.t,
        #         sigma=line_search_params.sigma,
        #     )
        # elif method_name == "simple_rule":
        #     lambdak, fstar, k = search(
        #         objfun=objfun,
        #         xk=xk,
        #         dk=dk,
        #         a0=line_search_params.a,
        #         b0=line_search_params.b,
        #         alpha0=line_search_params.alpha,
        #         rho=line_search_params.rho,
        #     )
        # else:
        #     phi = lambda alpha: objfun(xk + alpha * dk)

        #     # 使用进退法寻找包含极小值的区间
        #     init_alpha = (line_search_params.a + line_search_params.b) / 2.0
        #     (
        #         line_search_params.a,
        #         line_search_params.b,
        #         _,
        #     ) = chase(phi, init_alpha, h=line_search_params.h)

        #     lambdak, fstar, k = search(phi, line_search_params.a, line_search_params.b, line_search_params.epsilon)
        # return lambdak, fstar, k
