import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions
from NumericalOptimization.linear_search import LineSearchParams


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    objfun = functions.boha3

    x0 = jnp.array([80.0, -30.0])

    xstar, fstar, k = optimizer.gradient_methods.newton_goldstein(
        objfun, x0, epsilon=0.001, line_search_params=LineSearchParams(name="golden")
    )

    print(xstar, fstar, k)
