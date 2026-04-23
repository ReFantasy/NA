import jax
import jax.numpy as jnp
from NumericalOptimization.utils import linprog, linear_search
from loguru import logger


def feadesdir_lin(objfun, A, b, E, xk, gradfun=None, atol=1e-8):
    """
    使用 JAX 和 MPAX 确定可行下降方向
    """
    if gradfun == None:
        gradfun = jax.grad(objfun)
    grad_val = gradfun(xk)

    t = A @ xk
    active_mask = jnp.isclose(t, b, atol=atol)
    A1 = A[active_mask, :]
    if A1.size == 0:
        b = None
    else:
        b = jnp.zeros(A1.shape[0])

    if E.size == 0:
        beq = None
    else:
        beq = jnp.zeros(E.shape[0])

    d, _ = linprog(
        f=grad_val, A=A1, b=b, Aeq=E, beq=beq, lb=-1.0 * jnp.ones_like(grad_val), ub=1.0 * jnp.ones_like(grad_val)
    )
    return d, grad_val


def linesearch_lin(objfun, A, b, xk, d, atol=1e-8):
    # 确定积极约束矩阵A1
    t = A @ xk
    active_mask = jnp.isclose(t, b, atol=atol)
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

    # 设置一维搜索目标函数
    phi = lambda lam: objfun(xk + lam * d)

    final_lambda, _, _ = linear_search.golden(phi, a=0.0, b=lambda_max, epsilon=atol)
    return final_lambda


def zoutendijk_lin(objfun, A, b, E, x0, gradfun=None, max_iter=1000, atol: float = 1e-4, verbose=True):
    jnp.set_printoptions(formatter={"float": "{: .4e}".format})

    k = 0
    xk = x0

    while True:
        d, grad = feadesdir_lin(objfun, A, b, E, xk, gradfun=gradfun, atol=atol)

        if jnp.isclose(grad @ d, 0.0, atol=atol):
            if verbose:
                logger.info(
                    f"Iteration {k:4d}: x = {xk}, f(x) = {objfun(xk):.4e}, grad @ d = {grad @ d:.4e}, lambda = {lam:.4e}"
                )
            return xk, objfun(xk), k

        if k >= max_iter:
            logger.warning("Maximum iterations reached without convergence.")
            return xk, objfun(xk), k

        lam = linesearch_lin(objfun, A, b, xk, d, atol=atol**2)

        if verbose:
            logger.info(
                f"Iteration {k:4d}: x = {xk}, f(x) = {objfun(xk):.4e}, grad @ d = {grad @ d:.4e}, lambda = {lam:.4e}"
            )

        xk = xk + lam * d
        k += 1


if __name__ == "__main__":

    @jax.jit
    def objfun(x):
        return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] + 6.0

    @jax.jit
    def gradfun(x):
        return jnp.array([2 * x[0] - 2, 2 * x[1] - 4])

    A = jnp.array([[2.0, 1.0], [2.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([6.0, 0.0, 0.0, 0.0])
    E = jnp.array([])  # 没有等式约束
    e = jnp.array([])  # 没有等式约束的右侧值

    # x0 = jnp.array([1.0, 4.0])
    x0 = jnp.array([0.0, 4.0])

    xstar, fstar, iterations = zoutendijk_lin(
        objfun, A, b, E, x0, gradfun=None, verbose=True
    )  # gradfun=None: use jax auto diff
    print(f"Total iterations             : {iterations}")
    print(f"Optimal solution x           : {xstar}")
    print(f"Optimal objective value f(x) : {fstar}")
