def chase(objfun: callable, x_init: float, h: float):
    """
    使用进退法（Bounding Phase Method）寻找一元函数的极小值包含区间。
    作者: LONG QIANG (Created on Thu Mar 24 19:05:38 2022)
    通过不断步进试探，找到一个呈现“高-低-高”形态的区间，该区间必定包含极小值点。

    参数
    ----------
    func : callable
        目标函数。
    x_init : float
        初始搜索点。
    h : float
        初始搜索步长。

    返回
    -------
    tuple[float, float, int]
        包含极小值点的区间端点 x1, x3，以及迭代次数 k。
    """
    x1, x2 = x_init, x_init + h
    fx1, fx2 = objfun(x1), objfun(x2)
    k = 0

    if fx1 > fx2:
        while True:
            k += 1
            x3 = x2 + h
            fx3 = objfun(x3)
            if fx2 > fx3:
                x1, x2 = x2, x3
                fx1, fx2 = fx2, fx3
            else:
                return x1, x3, k
    else:
        x3, x2 = x2, x1
        fx3, fx2 = fx2, fx1
        while True:
            k += 1
            x1 = x2 - h
            fx1 = objfun(x1)
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
