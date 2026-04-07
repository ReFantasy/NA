import NumericalAnalysis as na
import jax.numpy as jnp

if __name__ == "__main__":
    import jax
    jax.config.update("jax_enable_x64", True)

    A = jnp.array([[5, 2, 1], [-1, 4, 2], [2, -3, 10]])
    b = jnp.array([-12, 20, 3])
    tol = 1e-6
    print("analytical solution: [-4 3 2]")

    x, info = na.LS.jacobi(A, b, tol)
    print(f"jacobi solution: {x} number of iter: {info.current_iter}")

    x, info = na.LS.seidel(A, b, tol)
    print(f"seidel solution: {x} number of iter: {info.current_iter}")

    x, info = na.LS.sor(A, b, w=0.8, tol=tol)
    print(f"sor    solution: {x} number of iter: {info.current_iter}")
