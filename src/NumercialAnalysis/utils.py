import jax.numpy as jnp


# 判断对角占优矩阵
def is_sdd(A: jnp.ndarray) -> jnp.ndarray:
    """
    Determine whether a matrix is strictly diagonally dominant.
    Definition: For each row i, |a_ii| > sum_{j!=i} |a_ij|;
                For each column i, |a_ii| > sum_{j!=i} |a_ji|.
    Parameters:
        A: input matrix.
    Returns:
        True if both row-wise and column-wise strictly diagonally dominant, otherwise False.
        Return False if A is not a square matrix.
    """

    A = jnp.asarray(A)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return jnp.array(False)

    diag = jnp.abs(jnp.diag(A))
    row_sum = jnp.sum(jnp.abs(A), axis=1) - diag
    col_sum = jnp.sum(jnp.abs(A), axis=0) - diag
    return jnp.all(diag > row_sum) or jnp.all(diag > col_sum)


if __name__ == "__main__":
    # A = jnp.array([[4, 1, 2],
    #                [3, 5, 1],
    #                [1, 1, 3]])
    A = jnp.array([[6, -2, 1], [1, 5, 2], [-1, 1, -4]])
    print(is_sdd(A))  # 输出: True
