"""
梯度类优化方法测试模块

Author:
    ReFantasy (Created on 2026-04-09 16:22:00)

Revisions:
    ## 2026-04-09 [ReFantasy]: ----
"""

from pytest import approx
from NumericalOptimization import gradient_methods
from NumericalOptimization import linear_search
import jax
import jax.numpy as jnp
import numpy as np
import NumericalOptimization
from NumericalOptimization.linear_search import LineSearchParams
from NumericalOptimization.utils import functions
from NumericalOptimization import utils

jax.config.update("jax_enable_x64", True)

linear_search_names = [
    "golden",
    "fibonacci",
    "newton",
    "secant",
    "parabola",
    "armijo_goldstein",
    "wolf_powell",
    "simple_rule",
]


# ---------------------------------------------------------------------
#                           最速梯度法
# ---------------------------------------------------------------------
class TestGradientDescent:

    @staticmethod
    def objfun(x):
        y = 4 * (x[0] - 2) ** 2 + 9 * (x[1] + 3) ** 2
        return y

    # 黄金分割法
    def test_gradient_descent(self):
        epsilon2 = 0.00001

        for name in linear_search_names:
            print("测试梯度下降法，线搜索方法：", name)
            linear_search_strategy = utils.LineSearchFunction(
                line_search_params=LineSearchParams(name=name, epsilon=epsilon2)
            )

            x0 = jnp.array([1.0, 1.0])
            epsilon = 0.001
            xstar, _, _ = gradient_methods.gradient_descent(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([2.0, -3.0], abs=0.0001)

            x0 = jnp.array([-2.0, 3.0])
            xstar, _, _ = gradient_methods.gradient_descent(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([2.0, -3.0], abs=0.0001)

            x0 = jnp.array([10.0, -10.0])
            xstar, _, _ = gradient_methods.gradient_descent(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([2.0, -3.0], abs=0.0001)


# ---------------------------------------------------------------------
#                           牛顿法
# ---------------------------------------------------------------------
class TestNewton:

    @staticmethod
    def objfun(x):
        y = x[0] ** 4 + x[0] * x[1] + (1 + x[1]) ** 2
        return y

    # 黄金分割法
    def test_newton(self):
        epsilon = 0.0000001
        for name in linear_search_names:
            print("测试牛顿法，线搜索方法：", name)
            linear_search_strategy = utils.LineSearchFunction(
                line_search_params=LineSearchParams(name=name, epsilon=epsilon)
            )

            x0 = jnp.array([1.0, 1.0])
            epsilon = 0.001
            xstar, _, _ = gradient_methods.newton_goldstein(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)

            x0 = jnp.array([-2.0, 3.0])
            xstar, _, _ = gradient_methods.newton_goldstein(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)

            x0 = jnp.array([10.0, -10.0])
            xstar, _, _ = gradient_methods.newton_goldstein(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)

        for name in linear_search_names:
            print("测试牛顿法，线搜索方法：", name)
            linear_search_strategy = utils.LineSearchFunction(
                line_search_params=LineSearchParams(name=name, epsilon=epsilon)
            )
            x0 = jnp.array([1.0, 1.0])
            epsilon = 0.001
            xstar, _, _ = gradient_methods.newton_goldfeld(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)

            x0 = jnp.array([-2.0, 3.0])
            xstar, _, _ = gradient_methods.newton_goldfeld(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)

            x0 = jnp.array([10.0, -10.0])
            xstar, _, _ = gradient_methods.newton_goldfeld(
                self.objfun, x0, epsilon, line_search_function=linear_search_strategy
            )
            assert np.array(xstar) == approx([0.6958, -1.3479], abs=0.001)


# ---------------------------------------------------------------------
#                           共轭梯度法
# ---------------------------------------------------------------------


class TestConjugateGradient:
    class LineSearchFunction:
        def __init__(self, line_search_params: LineSearchParams = LineSearchParams()):
            self.line_search_params = line_search_params

        def __call__(self, objfun, xk, dk):
            phi = lambda alpha: objfun(xk + alpha * dk)

            init_alpha = (self.line_search_params.a + self.line_search_params.b) / 2.0
            (
                self.line_search_params.a,
                self.line_search_params.b,
                _,
            ) = NumericalOptimization.utils.chase(phi, init_alpha, h=self.line_search_params.h)

            return linear_search.golden(
                phi=phi,
                a=self.line_search_params.a,
                b=self.line_search_params.b,
                epsilon=self.line_search_params.epsilon,
            )

    # 黄金分割法
    def test_conjugate_gradient(self):

        linear_search = self.LineSearchFunction(line_search_params=LineSearchParams(epsilon=0.0001))

        x0 = jnp.array([80.0, -30.0])

        xstar, fstar, k = gradient_methods.conjugate_gradient(
            functions.boha2,
            x0,
            epsilon=0.001,
            line_search_function=linear_search,
        )
        assert np.array(xstar) == approx([0.0, 0.0], abs=0.001)
