from pytest import approx
import NumericalOptimization.linear_search as linear_search
from NumericalOptimization.utils import chase

import math
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------
#                            追赶法测试
# ---------------------------------------------------------------------
def test_chase():
    def phi(x):
        y = x * math.sin(x)
        return y

    x0, h = -2.0, 1
    a, b, _ = chase(phi, x0, h)
    assert a == approx(-1.0)
    assert b == approx(1.0)

    x0, h = 5.0, 0.5
    a, b, _ = chase(phi, x0, h)
    assert a == approx(4.5)
    assert b == approx(5.5)

    x0, h = -13.0, 1
    a, b, _ = chase(phi, x0, h)
    assert a == approx(-12.0)
    assert b == approx(-10.0)


# ---------------------------------------------------------------------
#                            试探法测试
# ---------------------------------------------------------------------
class TestEliminationMethods:
    @staticmethod
    def phi(x):
        y = 2 * x**2 - 4 * x - 1
        return y

    # 黄金分割法
    def test_golden(self):
        a = -4.0  # 初始区间左端点
        b = 4.0  # 初始区间右端点
        epsilon = 0.1  # 容忍精度
        alpha, _, _ = linear_search.golden(self.phi, a, b, epsilon)
        assert alpha == approx(1.01685426, abs=1e-7)

    # 斐波那契法
    def test_fibonacci(self):
        a = -4.0  # 初始区间左端点
        b = 4.0  # 初始区间右端点
        epsilon = 0.1  # 容忍精度
        alpha, _, _ = linear_search.fibonacci(self.phi, a, b, epsilon)
        assert alpha == approx(1.07865168, abs=1e-7)


# ---------------------------------------------------------------------
#                            逼近法测试
# ---------------------------------------------------------------------
class TestApproximationMethods:
    @staticmethod
    def phi(x):
        y = math.e ** (-x) + x**2
        return y

    # 牛顿法
    def test_newton(self):
        a = -4.0  # 初始区间左端点
        b = 4.0  # 初始区间右端点
        epsilon = 0.00001  # 容忍精度
        alpha, _, _ = linear_search.newton(self.phi, a, b, epsilon)
        assert alpha == approx(0.35173371, abs=1e-7)

    # 割线法
    def test_secant(self):
        a = -4.0  # 初始区间左端点
        b = 4.0  # 初始区间右端点
        epsilon = 0.00001  # 容忍精度
        alpha, _, _ = linear_search.secant(self.phi, a, b, epsilon)
        assert alpha == approx(0.351733717, abs=1e-7)

    # 抛物线法
    def test_parabola(self):
        a = -4.0  # 初始区间左端点
        b = 4.0  # 初始区间右端点
        epsilon = 0.00001  # 容忍精度
        alpha, _, _ = linear_search.parabola(self.phi, a, b, epsilon)
        assert alpha == approx(0.35178535, abs=1e-7)


# ---------------------------------------------------------------------
#                            非精确一维搜索
# ---------------------------------------------------------------------
class TestInexactLineSearch:
    @staticmethod
    def objfun(x):
        y = (x[0] - 1) ** 2 + (x[1] + 1) ** 2
        return y

    def test_armijo_goldstein(self):
        xk = jnp.array([0.0, 0.0])  # 原目标函数当前跌点
        dk = -jax.grad(self.objfun)(xk)  # 当前搜索方向
        a0, b0, alpha0 = 0.0, 20.0, 10.0  # 初始区间和初始试探点
        alpha, _, _ = linear_search.armijo_goldstein(self.objfun, xk, dk, a0, b0, alpha0)
        assert alpha == approx(0.625, abs=1e-3)

    def test_wolf_powell(self):
        xk = jnp.array([0.0, 0.0])  # 原目标函数当前跌点
        dk = -jax.grad(self.objfun)(xk)  # 当前搜索方向
        a0, b0, alpha0 = 0.0, 20.0, 10.0  # 初始区间和初始试探点
        alpha, _, _ = linear_search.wolf_powell(self.objfun, xk, dk, a0, b0, alpha0)
        assert alpha == approx(0.625, abs=1e-3)

    def test_simple_rule(self):
        xk = jnp.array([0.0, 0.0])  # 原目标函数当前跌点
        dk = -jax.grad(self.objfun)(xk)  # 当前搜索方向
        a0, b0, alpha0 = 0.0, 20.0, 10.0  # 初始区间和初始试探点
        alpha, _, _ = linear_search.simple_rule(self.objfun, xk, dk, a0, b0, alpha0)
        assert alpha == approx(0.6461, abs=1e-3)
