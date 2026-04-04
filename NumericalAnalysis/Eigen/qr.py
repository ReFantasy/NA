import jax.numpy as jnp


def qr(A: jnp.array, tol: float = 1e-12, iter: int = 10000):
    for k in range(iter):
        Q, R = jnp.linalg.qr(A)
        A = R @ Q
        if jnp.all(jnp.abs(jnp.tril(A, -1)) < tol):
            break
    print(f"QR iteration converged in {k+1} iterations.")
    return jnp.diag(A)


if __name__ == "__main__":
    A = jnp.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])

    eigenvalues = qr(A, tol=1e-18)
    print("Eigenvalues:\n", eigenvalues)
