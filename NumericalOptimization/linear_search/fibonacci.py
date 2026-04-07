"""
斐波拉契法线搜索模块。

包含斐波拉契数列生成函数和一维线性搜索的斐波拉契算法实现。

Author:
    LONG QIANG (Created on Thu Mar 24 19:05:38 2022)

Revisions:
    2026-04-06 [ReFantasy]: 修改函数名；为 fibonacci 函数新增 objfun (callable) 参数以传入目标函数，方便外部调用和复用
"""


## 斐波拉契数列生成函数
def fibonacci_sequence(a, b, epsilon):
    bound = (b - a) / epsilon
    fibseq = [1.0, 1.0]
    while fibseq[-1] < bound:
        f = fibseq[-1] + fibseq[-2]
        fibseq.append(f)
    return fibseq, len(fibseq)


## 斐波拉契法线搜索
def fibonacci(objfun: callable, a: float, b: float, epsilon: float):
    """
    [精确线性搜索] 使用斐波那契搜索法寻找一元函数在给定区间 [a, b] 上的极小点。

    参数:
    ----------
        objfun (callable): 目标函数。
        a (float): 初始搜索区间的左端点。
        b (float): 初始搜索区间的右端点。
        epsilon (float): 容许误差控制精度。

    返回:
    ----------
        tuple[float, float, int]: 包含最优解 xstar、对应的函数值 fstar 以及迭代次数 k。
    """
    # 生成斐波拉契序列
    fibseq, n = fibonacci_sequence(a, b, epsilon)

    # 初始化
    ak = a  # 初始区间左端点
    bk = b  # 初始区间右端点
    k = 1  # 迭代计数器
    lambdak = ak + (fibseq[(n - 1) - 2] / fibseq[n - 1]) * (bk - ak)  # 左试探点
    muk = ak + (fibseq[(n - 1) - 1] / fibseq[n - 1]) * (bk - ak)  # 右试探点
    flambdak = objfun(lambdak)  # 左试探点函数值
    fmuk = objfun(muk)  # 右试探点函数值

    # 迭代过程
    while True:
        # print(k,ak,bk,bk-ak)
        if bk - ak <= epsilon:  # 终止条件判断
            xstar = (ak + bk) / 2
            fstar = objfun(xstar)
            return xstar, fstar, k
        else:
            k += 1  # 计数器加 1
            if flambdak < fmuk:  # 情形 1
                bk = muk
                muk = lambdak
                fmuk = flambdak
                lambdak = ak + (fibseq[(n - 1) - k - 1] / fibseq[(n - 1) - k + 1]) * (bk - ak)
                flambdak = objfun(lambdak)
            else:  # 情形 2
                ak = lambdak
                lambdak = muk
                flambdak = fmuk
                muk = ak + (fibseq[(n - 1) - k] / fibseq[(n - 1) - k + 1]) * (bk - ak)
                fmuk = objfun(muk)


if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def objfun(x):
        y = 2 * x**2 - 4 * x - 1
        return y

    # 输入
    a = -4.0  # 初始区间左端点
    b = 4.0  # 初始区间右端点
    epsilon = 0.1  # 容忍精度

    # 斐波拉契法
    xstar, fstar, k = fibonacci(objfun, a, b, epsilon)
    print("Fibonacci Method: ", xstar, fstar, k)
