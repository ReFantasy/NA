import NumericalOptimization.linear_search as linear_search


def line_search_function(objfun, xk, dk, method_name="golden", a=0.0, b=3.0, init_alpha=1.0, epsilon: float = 0.00001):
    """
    线搜索算法接口函数，根据指定的算法名称调用相应的线搜索方法。
    参数
    ----------
    objfun : callable
        目标函数。
    xk : array-like
        当前迭代点。
    dk : array-like
        当前搜索方向。
    method_name : str, optional
        线搜索方法名称，默认为 "golden"。可选值包括 "golden", "newton", "fibonacci", "armijo_goldstein", "wolf_powell", "secant", "parabola", "simple_rule"。
    a : float, optional
        初始区间左端点，默认为 0.0。
    b : float, optional
        初始区间右端点，默认为 3.0。
    init_alpha : float, optional
        初始试探点，默认为 1.0。
    epsilon : float, optional
        容忍精度，默认为 0.00001。

    Returns
    -------
    tuple[float, float, int]
        最优步长 lambdak，最优函数值 fstar，以及迭代次数 k。

    Notes
    -----
    该函数根据指定的线搜索方法名称调用相应的算法实现，并返回最优步长、最优函数值和迭代次数。对于 "wolf_powell"、 "armijo_goldstein" 、"simple_rule" 方法，需要传入原始的优化目标函数 objfun、当前迭代点 xk、搜索方向 dk，以及初始区间和试探点等参数。
    对于其他方法，需要显示构造一维函数 phi = lambda alpha: objfun(xk + alpha * dk) 来表示沿搜索方向 dk 的函数值，并调用相应的线搜索算法。
    """

    def line_search_method(name="golden"):
        if name == "golden":
            return linear_search.golden
        elif name == "newton":
            return linear_search.newton
        elif name == "fibonacci":
            return linear_search.fibonacci
        elif name == "armijo_goldstein":
            return linear_search.armijo_goldstein
        elif name == "wolf_powell":
            return linear_search.wolf_powell
        elif name == "secant":
            return linear_search.secant
        elif name == "parabola":
            return linear_search.parabola
        elif name == "simple_rule":
            return linear_search.simple_rule
        else:
            raise ValueError("Unknown line search method: {}".format(name))

    search = line_search_method(method_name)
    if method_name == "wolf_powell" or method_name == "armijo_goldstein" or method_name == "simple_rule":
        lambdak, fstar, k = search(objfun=objfun, xk=xk, dk=dk, a0=a, b0=b, alpha0=init_alpha)
    else:
        phi = lambda alpha: objfun(xk + alpha * dk)
        lambdak, fstar, k = search(phi, a, b, epsilon)
    return lambdak, fstar, k


def chase(phi: callable, x_init: float, h: float):
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
    fx1, fx2 = phi(x1), phi(x2)
    k = 0

    if fx1 > fx2:
        while True:
            k += 1
            x3 = x2 + h
            fx3 = phi(x3)
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
            fx1 = phi(x1)
            if fx2 < fx1:
                return x1, x3, k
            else:
                x3, x2 = x2, x1
                fx3, fx2 = fx2, fx1


