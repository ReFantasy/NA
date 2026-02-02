import jax.numpy as jnp
from NumercialAnalysis.LS import Info


def sor(A: jnp.array, b: jnp.array, w, tol, x0: jnp.array = None):
    # TODO 判断是否收敛

    U = jnp.triu(A)
    L = jnp.tril(A)
    D = A + L + U

    D_wL_inv = jnp.linalg.inv(D - w * L)
    Bw = D_wL_inv @ ((1.0 - w) * D + w * U)
    gw = w * D_wL_inv @ b

    if x0 == None:
        x0 = jnp.zeros(len(b))
    x_pre = x0
    x = Bw @ x_pre + gw
    info = Info(current_iter=1)
    while jnp.linalg.norm(x - x_pre, ord=jnp.inf) > tol:
        x_pre = x
        x = Bw @ x_pre + gw
        info.current_iter = info.current_iter + 1
        if info.current_iter >= info.max_iter:
            break
    return x, info
