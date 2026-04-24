import jax
import jax.numpy as jnp
from NumericalOptimization.utils import linear_search
from loguru import logger


def proj_mat(M, x):
    if M.size == 0:
        return jnp.eye(len(x))
    else:
        MMT_inv = jnp.linalg.inv(jnp.dot(M, M.T))
        P = jnp.eye(len(x)) - jnp.dot(jnp.dot(M.T, MMT_inv), M)
        return P


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


def rosen(objfun, A, b, E, x0, gradfun=None, max_iter=1000, atol: float = 1e-4, verbose=True):
    jnp.set_printoptions(formatter={"float": "{: .4e}".format})

    if gradfun == None:
        gradfun = jax.grad(objfun)

    if E is None or E.size == 0:
        E = jnp.array([[]]).reshape(0, A.shape[1])  # 如果 E 为空，则创建一个形状为 (0, n) 的空矩阵

    k = 0
    xk = x0

    while True:
        if k >= max_iter:
            logger.warning("Maximum iterations reached without convergence.")
            return xk, objfun(xk), k

        k += 1
        t = A @ xk
        active_mask = jnp.isclose(t, b, atol=atol)
        A1 = A[active_mask, :]  # 获取积极约束矩阵A1
        lam = 0.0

        while True:
            M = jnp.vstack([A1, E])
            Q = proj_mat(M, xk)  # 计算投影矩阵Q
            grad_val = gradfun(xk)
            d = -jnp.dot(Q, grad_val)  # 计算下降方向d
            if jnp.isclose(jnp.linalg.norm(d), 0.0, atol=atol):
                if M.size == 0:
                    xstar = xk
                    fstar = objfun(xstar)
                    if verbose:
                        logger.info(f"Iteration {k:4d}: x = {xk}, f(x) = {objfun(xk):.4e}, d = {d}, lambda = {lam:.4e}")
                    return xstar, fstar, k
                else:
                    MMT_inv = jnp.linalg.inv(jnp.dot(M, M.T))
                    w = -jnp.dot(jnp.dot(MMT_inv, M), grad_val)  # 计算拉格朗日乘子w
                    u = w[: A1.shape[0]]  # 获取对应于A1的部分
                    if jnp.all(u >= 0):
                        xstar = xk
                        fstar = objfun(xstar)
                        if verbose:
                            logger.info(
                                f"Iteration {k:4d}: x = {xk}, f(x) = {objfun(xk):.4e}, d = {d}, lambda = {lam:.4e}"
                            )
                        return xstar, fstar, k
                    else:
                        # 从A1中移除对应于u < 0的约束
                        A1 = A1[u >= 0, :]

            else:
                # 如果d为非零向量，则进行线搜索和迭代
                break

        # 线搜索
        lam = linesearch_lin(objfun, A, b, xk, d, atol=atol**2)

        if verbose:
            logger.info(f"Iteration {k:4d}: x = {xk}, f(x) = {objfun(xk):.4e}, d = {d}, lambda = {lam:.4e}")

        # 迭代
        xk = xk + lam * d


if __name__ == "__main__":

    def objfun(x):
        x1, x2 = x
        return 2.0 * x1**2 + 2.0 * x2**2 - 2.0 * x1 * x2 - 4.0 * x1 - 6.0 * x2

    A = jnp.array([[1.0, 1.0], [1.0, 5.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([2.0, 5.0, 0.0, 0.0])
    E = jnp.array([[]])
    x0 = jnp.array([0.2038, 0.7087])
    x, f, k = rosen(objfun, A, b, E, x0)
    print(f"Optimal solution: x = {x}, f(x) = {f}, iterations = {k}")
