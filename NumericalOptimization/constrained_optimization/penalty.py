import jax
import jax.numpy as jnp
from loguru import logger
from NumericalOptimization.gradient_methods import quasi_newton
from NumericalOptimization import utils


class ExtPenalty:
    def __init__(
        self,
        objfun,
        ineq_cons: list,
        eq_cons: list,
        x0: jnp.ndarray,
        epsilon=1e-4,
        max_iter=1000,
        sigma=1.0,
        c=1.5,
        line_search_function=utils.LineSearchFunction(),
        epsilon2=None,
    ):
        self.objfun = objfun
        self.ineq_cons = ineq_cons if ineq_cons is not None else []
        self.eq_cons = eq_cons if eq_cons is not None else []
        self.x0 = x0
        self.sigma = sigma
        self.c = c
        self.epsilon = epsilon
        self.epsilon2 = epsilon2 if epsilon2 is not None else epsilon * 0.1
        self.max_iter = max_iter
        self.line_search_function = line_search_function

    def init(self):
        # 构造外部罚函数
        @jax.jit
        def penalty_function(x):
            penalty = 0.0
            for g in self.ineq_cons:
                penalty += jnp.maximum(0.0, g(x)) ** 2
            for h in self.eq_cons:
                penalty += h(x) ** 2
            return penalty

        self.penalty_function = penalty_function

    def optimize(self, verbose=True):
        if getattr(self, "penalty_function", None) is None:
            self.init()

        jnp.set_printoptions(formatter={"float": "{: .4e}".format})

        k = 0
        while True:
            objfun = lambda x: self.objfun(x) + self.sigma * self.penalty_function(x)
            xstar, fstar, _ = quasi_newton(
                objfun=objfun,
                x0=self.x0,
                epsilon=self.epsilon2 * 0.1,
                gradfun=None,
                line_search_function=self.line_search_function,
            )
            k += 1

            if verbose:
                logger.info(
                    f"Iteration {k:4d}: x = {xstar}, f(x) = {fstar:.4e}, penalty = {self.penalty_function(xstar):.4e}, sigma = {self.sigma:.4e}, violation = {self.sigma * self.penalty_function(xstar):.4e}"
                )

            if self.sigma * self.penalty_function(xstar) < self.epsilon or k >= self.max_iter:
                break

            self.sigma *= self.c
            self.x0 = xstar

        return xstar, fstar, k


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    def objfun(x):
        x1, x2 = x
        return x1**2 + x2**2

    def h1(x):
        x1, x2 = x
        return x1 + x2 - 1.0

    eq_cons = [h1]
    ineq_cons = []
    x0 = jnp.array([0.0, 0.0])

    ext_pen = ExtPenalty(objfun, ineq_cons, eq_cons, x0)
    xstar, fstar, k = ext_pen.optimize(verbose=True)
    print(xstar, fstar, k)

    # ---------------------------------------------------------
    def objfun(x):
        x1, x2 = x
        return x1**2 + x2**2

    def g1(x):
        x1, x2 = x
        return 2 * x1 + x2 - 2.0

    def g2(x):
        x1, x2 = x
        return 1.0 - x2

    eq_cons = []
    ineq_cons = [g1, g2]
    x0 = jnp.array([3.0, 4.0])
    ext_pen = ExtPenalty(objfun, ineq_cons, eq_cons, x0)
    xstar, fstar, k = ext_pen.optimize()
    print(xstar, fstar, k)
