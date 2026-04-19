from math import isclose

import jax
import jax.numpy as jnp
from mpax import create_lp, r2HPDHG
from NumericalOptimization.utils import linear_search


def feadesdir(objfun, A, b, E, xk):
    """
    使用 JAX 和 MPAX 确定可行下降方向
    """
    grad = jax.grad(objfun)(xk)

    # 确定积极约束矩阵A1
    t = A @ xk
    active_mask = jnp.isclose(t, b)
    A_ub = A[active_mask, :]
    A_eq = E

    if A_ub.size == 0 and A_eq.size == 0:
        d = -jnp.sign(grad)
        return d, grad

    if A_ub.size == 0:
        A_ub = jnp.zeros(shape=(1, A.shape[1]))  # 没有不等式约束时，使用零矩阵
    if A_eq.size == 0:
        A_eq = jnp.zeros(shape=(1, A.shape[1]))  # 没有等式约束时，使用零矩阵

    b_ub = jnp.zeros(shape=(A_ub.shape[0],))
    b_eq = jnp.zeros(shape=(A_eq.shape[0],))

    # 调用 MPAX 求解线性规划
    # min  c^T x
    # s.t. Ax =  b
    #      Gx >= h
    lp = create_lp(
        c=grad,
        A=A_eq,
        b=b_eq,
        G=-A_ub,
        h=b_ub,
        l=-1.0,
        u=1.0,
        use_sparse_matrix=False,
    )
    solver = r2HPDHG(verbose=False)
    result = solver.optimize(lp)

    return result.primal_solution, grad


def linesearch(objfun, A, b, xk, d):
    # 确定积极约束矩阵A1
    t = A @ xk
    active_mask = jnp.isclose(t, b)
    # active_mask 取反，得到非积极约束矩阵A2
    inactive_mask = ~active_mask
    A2 = A[inactive_mask, :]
    b2 = b[inactive_mask]
    p = A2 @ d
    q = b2 - A2 @ xk

    # 求步长上界
    if jnp.all(p <= 0):
        lambda_max = 10.0
    else:
        # 获取 p > 0 的正数部分的索引
        mask_p_pos = p > 0
        q_pos = q[mask_p_pos]
        p_pos = p[mask_p_pos]

        lambda_max = jnp.min(q_pos / p_pos)
        # 如果 lambda_max 出现异常值（例如没提取到任何项），提供一个回退
        if jnp.isnan(lambda_max) or jnp.isinf(lambda_max):
            lambda_max = 10.0

    # 设置一维搜索目标函数
    phi = lambda lam: objfun(xk + lam * d)

    final_lambda, _, _ = linear_search.golden(phi, a=0.0, b=lambda_max, epsilon=0.00001)
    return final_lambda


def zoutendijk(objfun, A, b, E, e, x0, max_iter=1000):
    k = 0
    xk = x0

    while True:
        d, grad = feadesdir(objfun, A, b, E, xk)

        if jnp.isclose(grad @ d, 0.0):
            return xk, objfun(xk), k

        if k >= max_iter:
            print("Maximum iterations reached without convergence.")
            return xk, objfun(xk), k

        lam = linesearch(objfun, A, b, xk, d)
        xk = xk + lam * d
        k += 1


if __name__ == "__main__":

    @jax.jit
    def objfun(x):
        return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] + 6.0

    A = jnp.array([[2.0, 1.0], [2.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([6.0, 0.0, 0.0, 0.0])
    E = jnp.array([])  # 没有等式约束
    e = jnp.array([])  # 没有等式约束的右侧值
    # x0 = jnp.array([1.0, 4.0])
    x0 = jnp.array([0.4483, 4.0112])

    # d, grad = feadesdir(objfun, A, b, E, x0)
    # print("Feasible descent direction d:", d)
    # lam = linesearch(objfun, A, b, x0, d)
    # print("Optimal step size lambda:", lam)

    xstar, fstar, iterations = zoutendijk(objfun, A, b, E, e, x0)
    print("Optimal solution x:", xstar)
    print("Optimal objective value f(x):", fstar)
    print("Total iterations:", iterations)
