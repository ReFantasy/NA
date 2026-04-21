import jax
import jax.numpy as jnp
from NumericalOptimization.utils import linprog
from NumericalOptimization.utils import linear_search
from NumericalOptimization.utils.common import ConstraintFunctionSet


def feadesdir_nonlin(obj: callable, cons: ConstraintFunctionSet, xk: jnp.ndarray, atol=1e-8):
    n = len(xk)
    c_v, c_g = cons(xk)
    active_mask = jnp.isclose(c_v, jnp.zeros_like(c_v), atol=atol)
    dg = c_g[active_mask, :]
    dg = jnp.hstack(
        [dg, -jnp.ones((dg.shape[0], 1))]
    )  # 给 dg 矩阵的每一行末尾追加一列 -1.0，对应于 Zoutendijk 方法中辅助变量 z 的系数

    df = jax.grad(obj)(xk)
    df = jnp.append(df, -1.0).reshape(1, -1)

    A = jnp.vstack([df, dg])
    b = jnp.zeros((A.shape[0],))

    f = jnp.zeros(n)
    f = jnp.append(f, 1.0)  # 对应于 Zoutendijk 方法中辅助变量 z 的系数

    lb = -1.0 * jnp.ones(n)
    lb = jnp.append(lb, -jnp.inf)  # 对应于 Zoutendijk 方法中辅助变量 z 的下界
    ub = -lb
    Aeq = None
    beq = None
    dz, fdz = linprog(f=f, A=A, b=b, lb=lb, ub=ub, Aeq=Aeq, beq=beq)
    d = dz[:-1]  # 去掉最后一个元素，即辅助变量 z 的值
    return d, dz[-1]


def linesearch_nonlin(objfun, cons: ConstraintFunctionSet, xk: jnp.ndarray, d: jnp.ndarray, atol=1e-8):
    lambda_max = 1.0
    alpha = 1.1
    beta = 0.9

    if jnp.all(cons(xk + lambda_max * d)[0] <= 0):
        tmp_lam = lambda_max * alpha
        while jnp.all(cons(xk + tmp_lam * d)[0] <= 0):
            lambda_max = tmp_lam
            tmp_lam = lambda_max * alpha
    else:
        tmp_lam = lambda_max * beta
        lambda_max = tmp_lam
        while not jnp.all(cons(xk + tmp_lam * d)[0] <= 0):
            tmp_lam = tmp_lam * beta
            lambda_max = tmp_lam

    # 设置一维搜索目标函数
    phi = lambda lam: objfun(xk + lam * d)

    final_lambda, _, _ = linear_search.golden(phi, a=0.0, b=lambda_max, epsilon=atol)
    return final_lambda


def zoutendijk_nonlin(objfun, cons: ConstraintFunctionSet, x0: jnp.ndarray, max_iter=1000, atol=1e-4):
    k = 0
    xk = x0
    while True:
        d, z = feadesdir_nonlin(objfun, cons, xk, atol=atol)

        if jnp.isclose(z, 0.0, atol=atol):
            return xk, objfun(xk), k
        if k >= max_iter:
            print("Maximum iterations reached without convergence.")
            return xk, objfun(xk), k

        lambda_k = linesearch_nonlin(objfun, cons, xk, d, atol=atol)
        xk = xk + lambda_k * d
        k += 1
        # print(f"Iteration {k}: x = {xk}, f(x) = {objfun(xk)}")


if __name__ == "__main__":

    @jax.jit
    def objfun(x):
        x1, x2 = x
        return (x1 + 1) ** 2 + (x2 - 4) ** 2

    @jax.jit
    def constraint1(x):
        x1, x2 = x
        return x1**2 + x2**2 - 4

    @jax.jit
    def constraint2(x):
        x1, x2 = x
        return x1**2 - x2 - 1

    @jax.jit
    def constraint3(x):
        x1, x2 = x
        return -x1 + x2 - 2

    constraints = ConstraintFunctionSet(func_list=[constraint1, constraint2, constraint3])
    x = jnp.array([1.0, 0.0])
    # print("Constraint values at x:\n", np.array(constraints.evaluate(x)))
    # print("Constraint gradients at x:\n", np.array(constraints.gradient(x)))
    # v, g = constraints(x)
    # print("Constraint values at x:\n", np.array(v))
    # print("Constraint gradients at x:\n", np.array(g))

    # d, z = feadesdir(objfun, constraints, x)
    # print(d)
    # print(z)
    # lambda_val = linesearch(objfun, constraints, x, atol=1e-8)
    # print(lambda_val)

    xstar, fstar, iterations = zoutendijk_nonlin(objfun, constraints, x0=x)
    print(f"Total iterations             : {iterations}")
    print(f"Optimal solution x           : {xstar}")
    print(f"Optimal objective value f(x) : {fstar}")
