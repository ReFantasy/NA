from math import isclose

import jax
import jax.numpy as jnp
from mpax import create_lp, r2HPDHG
# from jax.experimental import sparse
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
    b_ub = jnp.zeros(shape=(A_ub.shape[0],))  # 右侧值为零，因为我们要满足 A1 d <= 0

    # 构造子规划问题的参数
    # 目标是最小化梯度方向的投影，即  ∇f(xk)^T d
    if E is not None and E.size > 0:
        A_eq = E
        b_eq = jnp.zeros(
            E.shape[0],
        )  # 等式约束右侧值为零，满足 E d = 0
    else:
        A_eq = jnp.zeros(shape=(1, A.shape[1]))  # 没有等式约束时，使用零矩阵
        b_eq = jnp.zeros(1)

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
        l=-1.0,#-jnp.ones_like(xk),
        u=1.0,#jnp.ones_like(xk),
        use_sparse_matrix=False,
    )  # 定义解的范围 d \in [l, u]
    solver = r2HPDHG(verbose=False, eps_abs=1e-4, eps_rel=1e-4,)
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
    # def phi(lam):
    #     return objfun(xk + lam * d)
    phi = lambda lam: objfun(xk + lam * d)
    
    
    final_lambda, fstar, k = linear_search.golden(phi, a = 0.0, b = lambda_max, epsilon=0.01)
    return final_lambda

def zoutendijk(objfun, A, b, E, e, x0, max_iter=1000):
    k= 0 
    xk = x0

    while True:
        d, grad = feadesdir(objfun, A, b, E, xk)

        if jnp.isclose(grad @ d, 0.0, atol=1e-3):
            #print("Optimal solution found at iteration", k)
            return xk, objfun(xk), k
        
        if k >= max_iter:
            print("Maximum iterations reached")
            return xk, objfun(xk), k

        lam = linesearch(objfun, A, b, xk, d)
        xk = xk + lam * d
        k += 1
        print(f"Iteration {k}: x = {xk}, f(x) = {objfun(xk)}, lambda = {lam}")

        
        


if __name__ == "__main__":

    @jax.jit
    def objfun(x):
        return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] + 6.0

    A = jnp.array([[2.0, 1.0], [2.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    #A = sparse.BCOO.fromdense(A)
    b = jnp.array([6.0, 0.0, 0.0, 0.0])
    E = jnp.array([])  # 没有等式约束
    e = jnp.array([])  # 没有等式约束的右侧值
    x0 = jnp.array([1.0, 4.0])
    #x0 = jnp.array([0.4483, 4.0112])

    d, grad = feadesdir(objfun, A, b, E, x0)
    print("Feasible descent direction d:", d)
    # lam = linesearch(objfun, A, b, x0, d)
    # print("Optimal step size lambda:", lam)

    # xstar, fstar, iterations = zoutendijk(objfun, A, b, E, e, x0)
    # print("Optimal solution x*:", xstar)
    # print("Optimal objective value f(x*):", fstar)
    # print("Total iterations:", iterations)
