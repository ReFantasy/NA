def step(func, xk):
    x_k1_1 = func(xk)
    x_k1_2 = func(x_k1_1)

    x_k1 = xk - (x_k1_1 - xk) ** 2 / (x_k1_2 - 2 * x_k1_1 + xk)
    return x_k1


def aitken(func, x0, tol):
    xk = x0
    x_k1 = step(func, xk)
    while abs(x_k1 - xk) > tol:
        xk = x_k1
        x_k1 = step(func, xk)
    return x_k1
