"""
The conjugate gradient method is an iterative method for solving  the system of linear equations
    A x = b
where A is symmetric and positive-definite.

The conjugate gradient method for solving the system of linear equations
is equivalent to the conjugate gradient method for solving the following optimization problem:
    min_x 0.5 x^T A x - b^T x,
which is a convex quadratic optimization problem.

ref: https://en.wikipedia.org/wiki/Conjugate_gradient_method
"""

import jax
import jax.numpy as jnp
from jax.experimental import sparse


def cg(A: sparse.BCOO, b: jnp.array, tol=1e-6, x0: jnp.array = None):
    if x0 == None:
        x = jnp.zeros(len(b))
    else:
        x = x0

    r = b - A @ x

    # 判断是否终止
    if jnp.linalg.norm(r, ord=jnp.inf) < tol:
        return x, r, k

    p = r
    k = 0
    while True:
        alpha_k = (r.T @ r) / (p.T @ A @ p)
        x = x + alpha_k * p

        r_old = r
        r = r - alpha_k * A @ p

        # 判断是否终止
        if jnp.linalg.norm(r, ord=jnp.inf) < tol:
            break

        beta_k = (r.T @ r) / (r_old.T @ r_old)
        p = r + beta_k * p
        k += 1
    return x, b - A @ x, k


def pcg(M: sparse.CSR, A: sparse.BCOO, b: jnp.array, tol=1e-6, x0: jnp.array = None):
    if x0 == None:
        x = jnp.zeros(len(b))
    else:
        x = x0

    L = jax.scipy.linalg.cholesky(M.todense(), lower=True)

    r = b - A @ x

    # z = jnp.linalg.solve(M, r)
    # z = sparse.linalg.spsolve(data=M.data, indices=M.indices, indptr=M.indptr, b=r)
    a = jnp.linalg.solve(L, r)
    z = jnp.linalg.solve(L.T, a)

    p = z
    k = 0
    while True:
        alpha_k = (r.T @ z) / (p.T @ A @ p)
        x = x + alpha_k * p
        r_old = r
        r = r - alpha_k * A @ p
        if jnp.linalg.norm(r, ord=jnp.inf) < tol:
            break
        z_old = z

        # z = jnp.linalg.solve(M, r)
        # z = sparse.linalg.spsolve(data=M.data, indices=M.indices, indptr=M.indptr, b=r)
        a = jnp.linalg.solve(L, r)
        z = jnp.linalg.solve(L.T, a)

        # Fletcher–Reeves formula: beta_k = (r.T @ z) / (r_old.T @ z_old)
        beta_k = (r.T @ (z - z_old)) / (r_old.T @ z_old)  # Polak–Ribière formula

        p = z + beta_k * p
        k += 1
    return x, b - A @ x, k


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    A = jnp.array(
        (
            [
                [6, 0, 1, 2, 0, 0, 2, 1],
                [0, 5, 1, 1, 0, 0, 3, 0],
                [1, 1, 6, 1, 2, 0, 1, 2],
                [2, 1, 1, 7, 1, 2, 1, 1],
                [0, 0, 2, 1, 6, 0, 2, 1],
                [0, 0, 0, 2, 0, 4, 1, 0],
                [2, 3, 1, 1, 2, 1, 5, 1],
                [1, 0, 2, 1, 1, 0, 1, 3],
            ]
        ),
        dtype=jnp.float64,
    )
    b = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float64)

    A_sp = sparse.BCOO.fromdense(A)
    # print(A_sp.data)
    # print(A_sp.indices)

    x, r, k = cg(A_sp, b)
    print(x)
    print(f"residual: {r.T @ r}, number of iterations: {k}")

    print("-----------------------------------------------")

    M = jnp.diag(jnp.diag(A))
    M = sparse.CSR.fromdense(M)
    x, r, k = pcg(M, A_sp, b)

    print(x)
    print(f"residual: {r.T @ r}, number of iterations: {k}")
