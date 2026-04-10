import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    objfun = functions.boha2

    x0 = jnp.array([93.2, -87.5])

    xstar, fstar, k = optimizer.gradient_methods.gradient_descent(
        objfun, x0, epsilon=1e-7, line_search_name="simple_rule"
    )
    print(xstar, fstar, k)

    xstar, fstar, k = optimizer.gradient_methods.newton_goldfeld(
        objfun, xstar, epsilon=1e-8, line_search_name="simple_rule"
    )

    print(xstar, fstar, k)
