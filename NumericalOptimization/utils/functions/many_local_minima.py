import jax
import jax.numpy as jnp

def ackley(xx, a=20.0, b=0.2, c=2.0 * jnp.pi):
    d = len(xx)

    sum1 = jnp.sum(xx ** 2)
    sum2 = jnp.sum(jnp.cos(c * xx))

    term1 = -a * jnp.exp(-b * jnp.sqrt(sum1 / d))
    term2 = -jnp.exp(sum2 / d)

    y = term1 + term2 + a + jnp.exp(1.0)
    return y

if __name__ == "__main__":
    import NumericalOptimization as optimizer
    from NumericalOptimization.utils import LineSearchFunction
    from NumericalOptimization.utils.draw import draw2d

    xstar, fstar, k = optimizer.gradient_methods.quasi_newton(
        ackley,
        jnp.array([10.0, 20.0]),
        epsilon=1e-6,
        line_search_function=LineSearchFunction(),
    )
    print(f"xstar: {xstar}, fstar: {fstar}, k: {k}")

    draw2d(ackley, x_range=(-40, 40), y_range=(-40, 40), samples_x=240, samples_y=240)