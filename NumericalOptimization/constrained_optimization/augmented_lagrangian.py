import jax
import jax.numpy as jnp
from loguru import logger
from NumericalOptimization import gradient_methods
from NumericalOptimization import utils
from NumericalOptimization import linear_search


class AugmentedLagrangianEq:
    def __init__(self, objfun: callable, eq_cons: list, x0: jnp.ndarray):
        self.objfun = objfun
        self.eq_cons = eq_cons
        self.x0 = x0

        self.n = len(x0)
        self.m = len(eq_cons)

        self.sigma = 1.0

    def __construct_augmented_lagrangian_function(self):
        @jax.jit
        def augmented_lagrangian_function(xv):
            x = xv[: self.n]
            v = xv[self.n :]
            penalty = 0.0
            for i, h in enumerate(self.eq_cons):
                penalty += v[i] * h(x)
                penalty += self.sigma / 2.0 * h(x) ** 2

            penalty += self.objfun(x)
            return penalty

        self.augmented_lagrangian_function = augmented_lagrangian_function

    def optimize(self, verbose=True):
        xk = self.x0
        v = jnp.ones(self.m)

        while True:
            self.__construct_augmented_lagrangian_function()  # 每次迭代都要更新目标函数，因为v和sigma在变化
            objfun = lambda xv: self.augmented_lagrangian_function(xv)
            xv = jnp.concatenate([xk, v])

            xvstar, fstar, _ = gradient_methods.newton_goldstein(
                objfun=objfun,
                x0=xv,
                epsilon=1e-3,
                gradfun=None,
                line_search_function=utils.LineSearchFunction(),
            )
            new_xk = xvstar[: self.n]
            v = xvstar[self.n :]

            hxk = jnp.array([h(xk) for h in self.eq_cons])
            hxk_new = jnp.array([h(new_xk) for h in self.eq_cons])
            if jnp.linalg.norm(hxk_new) < 1e-4:
                return new_xk, fstar

            if hxk_new / hxk > 0.75:
                self.sigma *= 1.5

            v = v + self.sigma * hxk_new
            xk = new_xk


class MyLineSearchFunction(utils.LineSearchFunction):
    def __init__(self, line_search_params: linear_search.LineSearchParams = None):
        self.line_search_params = line_search_params

    def __call__(self, objfun, xk, dk):

        return linear_search.armijo_goldstein(objfun=objfun, xk=xk, dk=dk)


class AugmentedLagrangian:
    def __init__(self, objfun: callable, ineq_cons: list, eq_cons: list, x0: jnp.ndarray):
        self.objfun = objfun
        self.ineq_cons = ineq_cons
        self.eq_cons = eq_cons
        self.x0 = x0

        self.n = len(x0)
        self.m = len(ineq_cons)
        self.l = len(eq_cons)

        self.sigma = 2.0

    def __construct_augmented_lagrangian_function(self):
        @jax.jit
        def augmented_lagrangian_function(xwv):
            x = xwv[: self.n]
            w = xwv[self.n : self.n + self.m]
            v = xwv[self.n + self.m :]

            penalty = 0.0
            for j, h in enumerate(self.eq_cons):
                penalty += v[j] * h(x)
                penalty += self.sigma / 2.0 * h(x) ** 2

            for i, g in enumerate(self.ineq_cons):
                term1 = jnp.maximum(0.0, w[i] + self.sigma * g(x)) ** 2
                term2 = w[i] ** 2
                penalty += (term1 - term2) / (2.0 * self.sigma)

            penalty += self.objfun(x)
            return penalty

        self.augmented_lagrangian_function = augmented_lagrangian_function

    def optimize(self, verbose=True):
        xk = self.x0
        w = jnp.ones(self.m)
        v = jnp.ones(self.l)

        while True:
            self.__construct_augmented_lagrangian_function()  # 每次迭代都要更新目标函数，因为v和sigma在变化
            objfun = lambda xwv: self.augmented_lagrangian_function(xwv)
            xwv = jnp.concatenate([xk, w, v])

            xwvstar, fstar, _ = gradient_methods.newton_goldstein(
                objfun=jax.jit(objfun),
                x0=xwv,
                epsilon=1e-4,
                gradfun=None,
                line_search_function=MyLineSearchFunction(),
            )
            new_xk = xwvstar[: self.n]
            w = xwvstar[self.n : self.n + self.m]
            v = xwvstar[self.n + self.m :]

            hxk = jnp.array([h(xk) for h in self.eq_cons])
            hxk_new = jnp.array([h(new_xk) for h in self.eq_cons])
            if jnp.linalg.norm(hxk_new) < 1e-4:
                return new_xk, fstar

            if hxk_new / hxk > 0.75:
                self.sigma *= 1.5

            w = jnp.maximum(0.0, w + self.sigma * jnp.array([g(new_xk) for g in self.ineq_cons]))
            v = v + self.sigma * hxk_new
            xk = new_xk


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    def objfun(x):
        x1, x2 = x
        return x1**2.0 + 2.0 * x2**2.0 - 2.0 * x1 * x2

    def constraint(x):
        x1, x2 = x
        return x1 + x2 - 1.0

    al = AugmentedLagrangian(objfun=objfun, ineq_cons=[], eq_cons=[constraint], x0=jnp.array([1.0, 2.0]))
    xk, fstar = al.optimize()
    print(xk, fstar)  # [0.6, 0.4]

    def objfun(x):
        x1, x2 = x
        return x1**2 + x2**2

    def constraint1(x):
        x1, x2 = x
        return x1**2 - 4 * x1 - x2 + 4.0

    al = AugmentedLagrangian(objfun=objfun, ineq_cons=[constraint1], eq_cons=[], x0=jnp.array([1.0, 2.0]))
    xk, fstar = al.optimize()
    print(xk, fstar)  # [1.12, 0.76] 1.85

    def objfun(x):
        x1, x2 = x
        return x1**2.0 + x2**2.0

    def constraint1(x):
        x1, x2 = x
        return x1 - x2 + 1.0

    al = AugmentedLagrangian(objfun=objfun, ineq_cons=[constraint1], eq_cons=[], x0=jnp.array([1.0, 2.0]))
    xk, fstar = al.optimize()
    print(xk, fstar)  # [-0.5, 0.5]
