import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions
from NumericalOptimization import linear_search
from NumericalOptimization import utils
from NumericalOptimization.linear_search import LineSearchParams


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    # 重载线搜索函数
    class LineSearchFunction(utils.LineSearchFunction):
        def __init__(self, line_search_params: LineSearchParams = LineSearchParams()):
            self.line_search_params = line_search_params

        def __call__(self, objfun, xk, dk):
            phi = lambda alpha: objfun(xk + alpha * dk)

            # return linear_search.simple_shrink(phi, alpha0=2.0, scaling=0.7)
            return linear_search.simple_sampled(phi, a=0.0, b=2.0, num_samples=100000000)

    search = LineSearchFunction(line_search_params=LineSearchParams(name=linear_search.types.golden, epsilon=0.0001))

    objfun = functions.boha2

    x0 = jnp.array([80.0, -30.0])

    xstar, fstar, k = optimizer.gradient_methods.newton_goldstein(
        objfun,
        x0,
        epsilon=0.0001,
        # line_search_function=utils.LineSearchFunction(
        #     line_search_params=LineSearchParams(name=linear_search.types.golden, epsilon=0.0001)
        # ),
        line_search_function=search,
    )
    print(xstar, fstar, k)
    while True:
        xstar, fstar, k = optimizer.gradient_methods.conjugate_gradient(
            objfun,
            x0,
            epsilon=0.0001,
            line_search_function=search,
        )

    print(xstar, fstar, k)
