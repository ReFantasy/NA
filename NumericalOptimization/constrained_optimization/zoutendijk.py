import jax
import jax.numpy as jnp
from mpax import create_lp, r2HPDHG
from jax.experimental import sparse


def feadesdir(fun, A, b, E, xk):
    """
    使用 JAX 和 MPAX 确定可行下降方向
    """
    val_and_grad_fn = jax.value_and_grad(fun)
    _, grad = val_and_grad_fn(xk)

    # 确定积极约束矩阵A1
    t = A @ xk  # jnp.dot(A, xk)
    active_mask = jnp.isclose(t, b)
    A_ub = A[active_mask, :]
    b_ub = jnp.zeros(shape=(A_ub.shape[0],))  # 右侧值为零，因为我们要满足 A1 d <= 0

    # 构造子规划问题的参数
    # 目标是最小化梯度方向的投影，即  ∇f(xk)^T d
    if E is not None and E.size > 0:
        A_eq = E
        b_eq = jnp.zeros(E.shape[0],)  # 等式约束右侧值为零，因为我们要满足 E d = 0
    else:
        A_eq = sparse.BCOO.fromdense(jnp.zeros(shape=(1, A.shape[1])))  # 没有等式约束时，使用零矩阵
        b_eq = jnp.zeros(1)

   

    # 调用 MPAX 求解线性规划
    # min c^T x
    # subject to Ax = b
    # .          Gx >= h
    
    # 定义界限 d \in [-1, 1]
    lb = -jnp.ones_like(xk)
    ub = jnp.ones_like(xk)

    lp = create_lp(
        c=grad,
        A = A_eq,
        b=b_eq,
        G= -A_ub,
        h=b_ub,
        l=lb,
        u=ub,
        use_sparse_matrix=False,
    )
    solver = r2HPDHG(verbose=False)
    result = solver.optimize(lp)
    #print("Optimal solution d:", result.primal_solution)
    return result.primal_solution


def zoutendijk(func, x0, max_iters=1000, tol=1e-6):
   
    pass


if __name__ == "__main__":
    @jax.jit
    def objfun(x):
        return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] + 6.0

    A = jnp.array([[2.0, 1.0], [2.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    A = sparse.BCOO.fromdense(A)
    b = jnp.array([6.0, 0.0, 0.0, 0.0])
    E = sparse.BCOO.fromdense(jnp.array([]))  # 没有等式约束
    e = jnp.array([])  # 没有等式约束的右侧值
    x0 = jnp.array([1.0, 4.0])
    
    d = feadesdir(objfun, A, b, E, x0)
    print("Feasible descent direction d:", d)
   

