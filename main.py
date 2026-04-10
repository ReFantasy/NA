import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    x0 = jnp.array([90.1, -80.3])

    xstar, fstar, k = optimizer.gradient_methods.gradient_descent(
        functions.boha1, x0, epsilon=1e-3, line_search_name="golden"
    )

    print(xstar, fstar, k)
