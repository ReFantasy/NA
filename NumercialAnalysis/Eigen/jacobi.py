import jax.numpy as jnp


def jacobi(A: jnp.array, tol: float = 1e-12):
    # 检查输入矩阵是否为实对称矩阵
    if not jnp.allclose(A, jnp.transpose(A)):
        raise ValueError("Input matrix must be symmetric.")

    # 创建单位矩阵
    Q = jnp.eye(A.shape[0])

    while True:
        # 找到最大非对角线元素
        max_val = 0
        p, q = 0, 1
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                if (abs(A[i, j]) > max_val) and (i != j):
                    max_val = abs(A[i, j])
                    p, q = i, j

        # 计算旋转角度
        theta_k = 0.5 * jnp.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

        # 创建旋转矩阵
        c = jnp.cos(theta_k)
        s = jnp.sin(theta_k)
        J = jnp.eye(A.shape[0])
        J = J.at[p, p].set(c).at[q, q].set(c).at[p, q].set(-s).at[q, p].set(s)

        # 更新矩阵和特征向量
        Q = Q @ J
        A = J.T @ A @ J
        # print(Q)
        # print("------------------------------")

        # 检查非对角元素的平方和
        off_diag_sum = jnp.sum(jnp.square(A - jnp.diag(jnp.diag(A))))
        if off_diag_sum < jnp.abs(tol):
            break

    return Q, jnp.diag(A)


def passby_jacobi(A: jnp.array, tol: float = 1e-12, scale: float = 0.5):
    # 检查输入矩阵是否为实对称矩阵
    if not jnp.allclose(A, jnp.transpose(A)):
        raise ValueError("Input matrix must be symmetric.")

    scale = jnp.abs(scale)
    scale = scale if scale < 1 else 0.5

    # 创建单位矩阵
    Q = jnp.eye(A.shape[0])

    alpha = jnp.sum(jnp.square(A - jnp.diag(jnp.diag(A)))) / A.shape[0]

    while alpha > tol:
        for i in range(A.shape[0]):
            for j in range(i + 1, A.shape[1]):
                if jnp.abs(A[i, j]) > alpha:
                    p, q = i, j
                    # 计算旋转角度
                    theta_k = 0.5 * jnp.arctan(2 * A[p, q] / (A[p, p] - A[q, q]))

                    # 创建旋转矩阵
                    c = jnp.cos(theta_k)
                    s = jnp.sin(theta_k)
                    J = jnp.eye(A.shape[0])
                    J = J.at[p, p].set(c).at[q, q].set(c).at[p, q].set(-s).at[q, p].set(s)
                    

                    # 更新矩阵和特征向量
                    Q = Q @ J
                    A = J.T @ A @ J
        alpha *= scale

    return Q, jnp.diag(A)
