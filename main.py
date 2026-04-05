import NumericalOptimization
from NumericalOptimization.utils import chase
import math
import jax
import jax.numpy as jnp


def main():
    print("Hello from na!")


@jax.jit
def objfun(x):
    return x * jnp.sin(x)


if __name__ == "__main__":

    x1, x3, k = chase(objfun, 5, 0.4)

    print(x1, x3, k)
