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
import numpy as np
import scipy
import scipy.sparse


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
        Ap = A @ p
        rdotr_old = r.T @ r

        alpha_k = rdotr_old / (p.T @ Ap)
        x = x + alpha_k * p
        r = r - alpha_k * Ap
        # 判断是否终止
        if jnp.linalg.norm(r, ord=jnp.inf) < tol:
            k += 1
            break
        beta_k = (r.T @ r) / rdotr_old
        p = r + beta_k * p
        k += 1
    return x, b - A @ x, k


def pcg(M: sparse.CSR, A: sparse.BCOO, b: jnp.array, tol=1e-6, x0: jnp.array = None):
    if x0 == None:
        x = jnp.zeros(len(b))
    else:
        x = x0

    r = b - A @ x

    L = jax.scipy.linalg.cholesky(M.todense(), lower=True)

    # z = jnp.linalg.solve(M, r)
    # z = sparse.linalg.spsolve(data=M.data, indices=M.indices, indptr=M.indptr, b=r)
    a = jax.scipy.linalg.solve_triangular(L, r, lower=True)
    z = jax.scipy.linalg.solve_triangular(L.T, a, lower=False)

    p = z
    k = 0
    while True:
        Ap = A @ p
        rz_old = r.T @ z

        alpha_k = rz_old / (p.T @ Ap)
        x = x + alpha_k * p
        r = r - alpha_k * Ap
        if jnp.linalg.norm(r, ord=jnp.inf) < tol:
            k += 1
            break

        # z = jnp.linalg.solve(M, r)
        # z = sparse.linalg.spsolve(data=M.data, indices=M.indices, indptr=M.indptr, b=r)
        a = jax.scipy.linalg.solve_triangular(L, r, lower=True)
        z = jax.scipy.linalg.solve_triangular(L.T, a, lower=False)

        beta_k = (r.T @ z) / rz_old  # Fletcher–Reeves formula:
        # beta_k = (r.T @ (z - z_old)) / rz_old  # Polak–Ribière formula

        p = z + beta_k * p
        k += 1
    return x, b - A @ x, k


if __name__ == "__main__":
    import time

    jax.config.update("jax_enable_x64", True)

    # A = jnp.array(
    #     (
    #         [
    #             [6, 0, 1, 2, 0, 0, 2, 1],
    #             [0, 5, 1, 1, 0, 0, 3, 0],
    #             [1, 1, 6, 1, 2, 0, 1, 2],
    #             [2, 1, 1, 7, 1, 2, 1, 1],
    #             [0, 0, 2, 1, 6, 0, 2, 1],
    #             [0, 0, 0, 2, 0, 4, 1, 0],
    #             [2, 3, 1, 1, 2, 1, 5, 1],
    #             [1, 0, 2, 1, 1, 0, 1, 3],
    #         ]
    #     ),
    #     dtype=jnp.float64,
    # )
    # b = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float64)

    # A = jnp.array(
    #     (
    #     [
    #         [60.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 1.0],
    #         [0.0, 50.0, 1.0, 1.0, 0.0, 0.0, 3.0, 0.0],
    #         [1.0, 1.0, 60.0, 1.0, 2.0, 0.0, 1.0, 2.0],
    #         [2.0, 1.0, 1.0, 70.0, 1.0, 2.0, 1.0, 1.0],
    #         [0.0, 0.0, 2.0, 1.0, 60.0, 0.0, 2.0, 1.0],
    #         [0.0, 0.0, 0.0, 2.0, 0.0, 40.0, 1.0, 0.0],
    #         [2.0, 3.0, 1.0, 1.0, 2.0, 1.0, 50.0, 1.0],
    #         [1.0, 0.0, 2.0, 1.0, 1.0, 0.0, 1.0, 30.0],
    #     ]
    #     ),
    #     dtype=jnp.float64,
    # )
    # b = jnp.array([1, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.float64)

    def create_2d_poisson_matrix(N):
        """
        生成 2D Poisson 方程离散化对应的系数矩阵（大小为 N^2 x N^2）。
        该矩阵为标准的大型稀疏对称正定矩阵。
        """
        # 1D 第二阶差分矩阵 (对角线为2, 次对角线为-1)
        diagonals = [-1.0 * np.ones(N - 1), 2.0 * np.ones(N), -1.0 * np.ones(N - 1)]
        T1 = scipy.sparse.diags(diagonals, offsets=[-1, 0, 1], format="csr")

        # 2D 差分矩阵可以通过 Kronecker 和 (Kronecker sum) 构建
        # A = T1 ⊗ I + I ⊗ T1
        I = scipy.sparse.eye(N, format="csr")
        A = scipy.sparse.kron(T1, I) + scipy.sparse.kron(I, T1)

        return A

    def create_random_spd_matrix(N, density=0.01, random_state=42):
        """
        生成一个通用的随机稀疏对称正定 (SPD) 矩阵。
        N: 矩阵维度 (N x N)
        density: 稀疏度，非零元素的大致比例
        """
        np.random.seed(random_state)

        # 1. 生成一个随机的稀疏矩阵 M
        M = scipy.sparse.random(
            N, N, density=density, format="csr", data_rvs=np.random.randn, random_state=random_state
        )

        M = M.todense()  # 转换为密集矩阵以便后续操作
        for i in range(N):
            M[i, i] = np.random.rand() * 10 + 1.0  # 确保对角线元素较大，增强正定性
        M = scipy.sparse.csr_matrix(M)  # 转回稀疏格式

        # 2. 构造 A = M * M^T，这保证了其对称半正定
        A = M.dot(M.T)

        # 3. 加上一个对角线偏移(Shift)以保证其严格正定，并控制条件数
        # 偏移量越大，条件数越小，残差收敛越快；偏移量越小，矩阵越病态
        shift = 0.1 * scipy.sparse.eye(N, format="csr")
        A = A + shift

        return A

    # ================= 测试大型矩阵 =================
    N_grid = 1000  # 网格大小，N=50 时矩阵大小为 2500 x 2500
    A_large = create_random_spd_matrix(N_grid, density=0.2, random_state=44)
    b_large = np.ones(A_large.shape[0])
    A = A_large.todense()
    # b = jnp.array(b_large, dtype=jnp.float64)
    b_random = np.random.randn(N_grid)
    b = jnp.array(b_random, dtype=jnp.float64)

    A_sp = sparse.BCOO.fromdense(A)

    t1 = time.time()
    x, r, k = cg(A_sp, b, tol=1e-4)
    t2 = time.time()
    print(f"CG method took {t2 - t1:.4f} seconds.")
    print(f"x: {x[:10]}...")  # 打印前10个元素以验证结果
    print(f"residual: {r.T @ r}, number of iterations: {k}")

    print("-----------------------------------------------")
    import eigenpy

    # M = jnp.diag(jnp.diag(A))
    # M = sparse.CSR.fromdense(M)

    ic = eigenpy.solvers.IncompleteCholesky()
    ic.compute(scipy.sparse.csc_matrix(A))
    L = ic.matrixL().todense()
    M = L @ L.T
    M = sparse.CSR.fromdense(M)

    t1 = time.time()
    x, r, k = pcg(M, A_sp, b, tol=1e-4)
    t2 = time.time()
    print(f"PCG method took {t2 - t1:.4f} seconds.")
    print(f"x: {x[:10]}...")  # 打印前10个元素以验证结果
    print(f"residual: {r.T @ r}, number of iterations: {k}")
