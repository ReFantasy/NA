from jax import grad


## 割线法
def secant(objfun: callable, a: float, b: float, epsilon: float, gradfun=None):
    # automatically compute the gradient of the objective function
    if gradfun == None:
        gradfun = grad(objfun)

    xk_1, xk = a, b
    k = 0

    while True:
        k += 1
        if abs(gradfun(xk)) <= epsilon:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            temp = xk
            xk -= ((xk - xk_1) / (gradfun(xk) - gradfun(xk_1))) * gradfun(xk)  # 式（2-49）
            xk_1 = temp


if __name__ == "__main__":
    import math
    import jax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def objfun(x):
        y = math.e ** (-x) + x**2
        return y

    # 输入
    a = -4.0  # 初始区间左端点
    b = 4.0  # 初始区间右端点
    epsilon = 0.00001  # 容忍精度

    # 割线法
    xstar, fstar, k = secant(objfun, a, b, epsilon)
    print("Secant: ", xstar, fstar, k)
