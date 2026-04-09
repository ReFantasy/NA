import jax.numpy as jnp


## 抛物线法
def parabola(phi: callable, a: float, b: float, epsilon: float):
    x1prime, x2prime, x3prime = a, (a + b) / 2.0, b
    fx1prime, fx2prime, fx3prime = phi(x1prime), phi(x2prime), phi(x3prime)
    xk, fxk = x2prime, fx2prime
    k = 0

    while True:
        k += 1
        a0 = jnp.array([[1, 1, 1]]).T
        a1 = jnp.array([[x1prime, x2prime, x3prime]]).T
        a2 = jnp.array([[x1prime**2, x2prime**2, x3prime**2]]).T
        b = jnp.array([[fx1prime, fx2prime, fx3prime]]).T
        D1_mat = jnp.concatenate((a0, b, a2), axis=1)
        D2_mat = jnp.concatenate((a0, a1, b), axis=1)
        D1_det = jnp.linalg.det(D1_mat)
        D2_det = jnp.linalg.det(D2_mat)
        xstarprime = -D1_det / (2.0 * D2_det)  # 式（2-55）

        xk_1, fxk_1 = xk, fxk
        xk, fxk = xstarprime, phi(xstarprime)

        if abs(fxk - fxk_1) < epsilon or abs(xk - xk_1) < epsilon:
            xstar = xk
            fstar = fxk
            return xstar, fstar, k
        else:
            x = jnp.array([x1prime, x2prime, x3prime, xk])
            f = jnp.array([fx1prime, fx2prime, fx3prime, fxk])
            index = jnp.argsort(x)
            xsort = x[index]
            fsort = f[index]
            fmin_index = jnp.argmin(fsort)  # 式（2-56）
            index = jnp.array([fmin_index - 1, fmin_index, fmin_index + 1])
            x1prime, x2prime, x3prime = xsort[index]
            fx1prime, fx2prime, fx3prime = fsort[index]
