def bisection(I,func, tol=1e-5):
    a, b = I
    x = (a + b) / 2
    value = func(x)

    if value == 0:
        return x

    while True:
        if func(a) * value < 0:
            b = x
        else:
            a = x
        x_n = (a + b) / 2
        if abs(x_n - x) < tol:
            return x_n
        x = x_n
        value = func(x)


