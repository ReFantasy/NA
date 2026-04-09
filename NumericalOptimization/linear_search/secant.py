from jax import grad


## 割线法
def secant(phi: callable, a: float, b: float, epsilon: float, gradfun=None):
    # automatically compute the gradient of the objective function
    if gradfun == None:
        gradfun = grad(phi)

    xk_1, xk = a, b
    k = 0

    while True:
        k += 1
        if abs(gradfun(xk)) <= epsilon:
            xstar = xk
            fstar = phi(xk)
            return xstar, fstar, k
        else:
            temp = xk
            xk -= ((xk - xk_1) / (gradfun(xk) - gradfun(xk_1))) * gradfun(xk)  # 式（2-49）
            xk_1 = temp
