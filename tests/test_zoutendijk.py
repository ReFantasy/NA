import time
import jax
import jax.numpy as jnp
from pytest import approx
from NumericalOptimization.utils import ConstraintFunctionSet
from NumericalOptimization.constrained_optimization import zoutendijk_lin, zoutendijk_nonlin
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_zoutendijk_lin():
    """
    线性约束优化问题示例：
    min f(x1,x2) = x1^2 + x2^2 - 2*x1 - 4*x2 + 6
    s.t. g1(x) = 2*x1 + x2 <= 6
         g2(x) = 2*x1 - x2 <= 0
         g3(x) = -x1 <= 0
         g4(x) = -x2 <= 0
    """

    @jax.jit
    def objfun(x):
        return x[0] ** 2 + x[1] ** 2 - 2 * x[0] - 4 * x[1] + 6.0

    @jax.jit
    def gradfun(x):
        return jnp.array([2 * x[0] - 2, 2 * x[1] - 4])

    A = jnp.array([[2.0, 1.0], [2.0, -1.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([6.0, 0.0, 0.0, 0.0])
    E = jnp.array([])  # 没有等式约束
    e = jnp.array([])  # 没有等式约束的右侧值

    # x0 = jnp.array([1.0, 4.0])
    x0 = jnp.array([0.0, 4.0])

    xstar, fstar, iterations = zoutendijk_lin(
        objfun, A, b, E, x0, gradfun=gradfun, verbose=True
    )  # gradfun=None: use jax auto diff
    assert np.array(xstar) == approx([1.0, 2.0], abs=0.001)
    assert fstar == approx(1.0, abs=0.001)


def test_zoutendijk_nonlin():
    """
    非线性约束优化问题示例：
    min f(x1,x2) = (x1 + 1)^2 + (x2 - 4)^2
    s.t. g1(x) = x1^2 + x2^2 - 4 <= 0
         g2(x) = x1^2 - x2 - 1 <= 0
         g3(x) = -x1 + x2 - 2 <= 0
    """

    @jax.jit
    def objfun(x):
        x1, x2 = x
        return (x1 + 1) ** 2 + (x2 - 4) ** 2

    @jax.jit
    def constraint1(x):
        x1, x2 = x
        return x1**2 + x2**2 - 4

    @jax.jit
    def constraint2(x):
        x1, x2 = x
        return x1**2 - x2 - 1

    @jax.jit
    def constraint3(x):
        x1, x2 = x
        return -x1 + x2 - 2

    constraints = ConstraintFunctionSet(func_list=[constraint1, constraint2, constraint3])
    x = jnp.array([1.0, 0.0])
    xstar, fstar, iterations = zoutendijk_nonlin(objfun, constraints, x0=x)
    assert np.array(xstar) == approx([0.0, 2.0], abs=0.001)
    assert fstar == approx(5.0, abs=0.001)
