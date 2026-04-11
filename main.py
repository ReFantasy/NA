import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions
from NumericalOptimization import linear_search


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    objfun = functions.boha2

    x0 = jnp.array([80.0, -30.0])

    xstar, fstar, k = optimizer.gradient_methods.newton_goldstein(
        objfun,
        x0,
        epsilon=0.001,
        line_search_params=linear_search.LineSearchParams(name=linear_search.types.golden, epsilon=0.0001),
    )
    print(xstar, fstar, k)

    xstar, fstar, k = optimizer.gradient_methods.conjugate_gradient(
        objfun,
        x0,
        epsilon=0.001,
        line_search_params=linear_search.LineSearchParams(name=linear_search.types.golden, epsilon=0.0001),
    )

    print(xstar, fstar, k)
