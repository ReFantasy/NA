from math import log
import string
import jax.numpy as jnp
from dataclasses import dataclass
import NumercialAnalysis as na
from loguru import logger

# from typing import Optional


@dataclass
class Info:
    current_iter: int = -1
    max_iter = 10000
    # height: float = 1.75  # 默认值
    # email: Optional[str] = None  # 可选字段

    # 可以定义方法
    # def is_adult(self) -> bool:
    #     return self.age >= 18


def is_converged(A: jnp.ndarray, B: jnp.ndarray, method: string = "Jacobi") -> bool:
    if jnp.linalg.norm(B, ord=jnp.inf) < 1:
        return True
    if jnp.linalg.norm(B, ord=1) < 1:
        return True
    if jnp.linalg.norm(B, ord=2) < 1:
        return True
    if jnp.linalg.norm(B, ord="f") < 1:
        return True
    if method == "Jacobi":
        if na.is_sdd(A):
            return True
    if method == "Seidel":
        if na.is_sdd(A):
            return True
    logger.warning("{} method may not converge!", method)
    return False
