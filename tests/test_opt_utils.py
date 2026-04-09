"""
优化工具测试
"""

import time
import jax
import jax.numpy as jnp

from NumericalOptimization.utils import is_pd, proj_pd

jax.config.update("jax_enable_x64", True)


def test_proj_pd():
    dim = 12
    # 随机生成一个非正定矩阵
    seed = time.time_ns()
    key = jax.random.key(seed)
    A = jax.random.normal(key, (dim, dim))
    while is_pd(A):
        key, subkey = jax.random.split(key)
        A = jax.random.normal(subkey, (dim, dim))

    assert is_pd(A) == False, "A 应该是非正定的"

    # 将 A 投影到正定矩阵空间
    A_proj = proj_pd(A)
    assert is_pd(A_proj) == True, "A_proj 应该是正定的"

    # 验证 A_proj 的特征值都大于等于 0
    eigvals = jnp.linalg.eigvalsh(A_proj)
    assert jnp.all(eigvals >= 0), "A_proj 的特征值应该都大于等于 0"

    # 使用随机向量验证 A_proj 的二次型是非负的
    ispd = True
    for _ in range(10000):
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, (dim,))
        qf = x.T @ A_proj @ x
        if qf <= 0:
            ispd = False
            break
    assert ispd == True, "A_proj 的二次型应该是非负的"
