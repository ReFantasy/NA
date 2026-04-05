def chase(f, x_init, h):
    x1, x2 = x_init, x_init + h
    fx1, fx2 = f(x1), f(x2)
    k = 0

    if fx1 > fx2:
        while True:
            k += 1
            x3 = x2 + h
            fx3 = f(x3)
            if fx2 > fx3:
                x1, x2 = x2, x3
            else:
                return x1, x3, k
    else:
        x3, x2 = x2, x1
        fx3, fx2 = fx2, fx1
        while True:
            k += 1
            x1 = x2 - h
            fx1 = f(x1)
            if fx2 < fx1:
                return x1, x3, k
            else:
                x3, x2 = x2, x1
                fx3, fx2 = fx2, fx1


if __name__ == "__main__":
    import jax
    import jax.numpy as jnp

    @jax.jit
    def objfun(x):
        return x * jnp.sin(x)

    # 第1组
    x0, h = -2.0, 1  # 初始试探点和试探步长
    a, b, k = chase(objfun, x0, h)  # 追赶法
    print(f"Input:x0={x0},h={h}; Output: [{a},{b}], {k}")

    # 第2组
    x0, h = 5.0, 0.5  # 初始试探点和试探步长
    a, b, k = chase(objfun, x0, h)  # 追赶法
    print(f"Input:x0={x0},h={h}; Output: [{a},{b}], {k}")

    # 第3组
    x0, h = -13.0, 1  # 初始试探点和试探步长
    a, b, k = chase(objfun, x0, h)  # 追赶法
    print(f"Input:x0={x0},h={h}; Output: [{a},{b}], {k}")
