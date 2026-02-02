import jax.numpy as jnp
from NumercialAnalysis.LS import Info


def seidel(A: jnp.array, b: jnp.array, tol, x0: jnp.array = None):
    # TODO 判断是否收敛

    U = jnp.triu(A)
    L = jnp.tril(A)
    D = A + L + U

    D_L_inv = jnp.linalg.inv(D - L)
    BS = D_L_inv @ U
    gS = D_L_inv @ b

    if x0 == None:
        x0 = jnp.zeros(len(b))

    x_pre = x0
    x = BS @ x_pre + gS

    info = Info(current_iter=1)
    while jnp.linalg.norm(x - x_pre, ord=jnp.inf) > tol:
        x_pre = x
        x = BS @ x_pre + gS
        info.current_iter = info.current_iter + 1
        if info.current_iter >= info.max_iter:
            break
    return x, info
