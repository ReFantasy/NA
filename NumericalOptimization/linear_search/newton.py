from jax import grad, hessian


## 牛顿法
def newton(objfun: callable, a: float, b: float, epsilon: float):
    # automatically compute the gradient and hessian of the objective function
    gradfun = grad(objfun)
    hessianfun = hessian(objfun)

    xk = (a + b) / 2
    k = 0

    while True:
        k += 1
        if abs(gradfun(xk)) <= epsilon:
            xstar = xk
            fstar = objfun(xk)
            return xstar, fstar, k
        else:
            xk -= gradfun(xk) / hessianfun(xk)  # 式（2-43）


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

    # 牛顿法
    xstar, fstar, k = newton(objfun, a, b, epsilon)
    print("Newton: ", xstar, fstar, k)
