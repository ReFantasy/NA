import jax
import jax.numpy as jnp
from pytest import approx
from NumericalOptimization.utils import ConstraintFunctionSet
from NumericalOptimization.constrained_optimization import rosen
import numpy as np

jax.config.update("jax_enable_x64", True)


def test_rosen():
    def objfun(x):
        x1, x2 = x
        return 2.0 * x1**2 + 2.0 * x2**2 - 2.0 * x1 * x2 - 4.0 * x1 - 6.0 * x2

    A = jnp.array([[1.0, 1.0], [1.0, 5.0], [-1.0, 0.0], [0.0, -1.0]])
    b = jnp.array([2.0, 5.0, 0.0, 0.0])
    E = jnp.array([[]])
    x0 = jnp.array([0.2038, 0.7087])
    x, f, _ = rosen(objfun, A, b, E, x0, atol=1e-6)
    assert np.array(x) == approx([1.129, 0.774], abs=0.001)
    assert f == approx(-7.16, abs=0.01)
