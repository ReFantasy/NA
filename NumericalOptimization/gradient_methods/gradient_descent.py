from NumericalOptimization import utils
import jax
import jax.numpy as jnp


## 最速下降法
def gradient_descent(objfun, x0, epsilon, line_search_name="golden"):
    gradfun = jax.grad(objfun)

    xk = x0
    k = 0
    while True:
        k += 1
        gk = gradfun(xk)
        if jnp.linalg.norm(gk) <= epsilon:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            dk = -gk

            # linear search
            lambdak, _, _ = utils.line_search_function(objfun, xk, dk, line_search_name)

            xk += lambdak * dk


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)

    ## 原问题目标函数
    def objfun(x):
        y = 4 * (x[0] - 2) ** 2 + 9 * (x[1] + 3) ** 2
        return y

    epsilon = 0.001

    line_search_name = "wolf_powell"  # "wolf_powell" "golden"

    ## 第1组
    x0 = jnp.array([1.0, 1.0])
    xstar, fstar, k = gradient_descent(objfun, x0, epsilon, line_search_name)
    print("第1组：")
    print("Input：x0 = {}".format(x0))
    print("Output: (xstar,fstar,k) = {}".format((xstar, fstar, k)))

    ## 第2组
    x0 = jnp.array([-2.0, 3.0])
    xstar, fstar, k = gradient_descent(objfun, x0, epsilon, line_search_name)
    print("第2组：")
    print("Input：x0 = {}".format(x0))
    print("Output: (xstar,fstar,k) = {}".format((xstar, fstar, k)))

    ## 第3组
    x0 = jnp.array([10.0, -10.0])
    xstar, fstar, k = gradient_descent(objfun, x0, epsilon, line_search_name)
    print("第3组：")
    print("Input：x0 = {}".format(x0))
    print("Output: (xstar,fstar,k) = {}".format((xstar, fstar, k)))

    # 第1组：
    # Input：x0 = [1. 1.]
    # Output: (xstar,fstar,k) = (array([ 1.99997618, -3.00000187]), np.float64(2.3005343989825326e-09), 6)
    # 第2组：
    # Input：x0 = [-2.  3.]
    # Output: (xstar,fstar,k) = (array([ 1.99998104, -2.99997138]), np.float64(8.81133832674739e-09), 9)
    # 第3组：
    # Input：x0 = [ 10. -10.]
    # Output: (xstar,fstar,k) = (array([ 2.00004392, -2.9999901 ]), np.float64(8.597280705350136e-09), 12)
