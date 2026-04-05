## 黄金分割法
def Golden(objfun: callable, a: float, b: float, epsilon: float) -> tuple[float, float, int]:
    # 初始化
    ak = a  # 初始区间左端点
    bk = b  # 初始区间右端点
    k = 1  # 迭代计数器
    lambdak = ak + 0.382 * (bk - ak)  # 左试探点
    muk = ak + 0.618 * (bk - ak)  # 右试探点
    flambdak = objfun(lambdak)  # 左试探点函数值
    fmuk = objfun(muk)  # 右试探点函数值

    # 迭代过程
    while True:
        # print(k, ak, bk, bk - ak)
        if bk - ak <= epsilon:  # 终止条件判断
            xstar = (ak + bk) / 2
            fstar = objfun(xstar)
            return xstar, fstar, k
        else:
            k += 1  # 迭代计数
            if flambdak < fmuk:  # 情形 1
                bk = muk
                muk = lambdak
                fmuk = flambdak
                lambdak = ak + 0.382 * (bk - ak)
                flambdak = objfun(lambdak)
            else:  # 情形 2
                ak = lambdak
                lambdak = muk
                flambdak = fmuk
                muk = ak + 0.618 * (bk - ak)
                fmuk = objfun(muk)


if __name__ == "__main__":
    import jax

    @jax.jit
    def objfun(x):
        y = 2 * x**2 - 4 * x - 1
        return y

    # 输入
    a = -4.0  # 初始区间左端点
    b = 4.0  # 初始区间右端点
    epsilon = 0.1  # 容忍精度

    # 黄金分割法
    xstar, fstar, k = Golden(objfun, a, b, epsilon)
    print("Golden Method: ", xstar, fstar, k)
