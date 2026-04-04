def bisection(I, func, tol=1e-5, max_iter=None):
    cur_iter = 0

    a, b = I
    x = (a + b) / 2
    val = func(x)

    if val == 0:
        return x

    while True:
        if func(a) * val < 0:
            b = x
        else:
            a = x
        x_n = (a + b) / 2

        val = func(x_n)
        if (abs(x_n - x) < tol) or (val == 0):
            return x_n

        # Check for maximum iterations
        cur_iter += 1
        if max_iter is not None:
            if cur_iter >= max_iter:
                return x_n
        x = x_n
