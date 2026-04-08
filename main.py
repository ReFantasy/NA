import NumericalOptimization
import NumericalOptimization as optimizer
import jax


def main():
    print("Hello from na!")


@jax.jit
def objfun(x):
    # return x * jnp.sin(x)
    return (x - 1) ** 2 + 1.5


if __name__ == "__main__":

    x1, x3, k = NumericalOptimization.utils.chase(objfun, -0.6, 0.4)

    print(x1, x3, k)
    x_star, y_star, _ = optimizer.linear_search.golden(objfun, x1, x3, 1e-8)
    print(x_star, y_star)

    NumericalOptimization.linear_search.armijo_goldstein(objfun, x1, -0.5, 0.0, 20.0, 10.0)
    NumericalOptimization.linear_search.wolf_powell(objfun, x1, -0.5, 0.0, 20.0, 10.0)
