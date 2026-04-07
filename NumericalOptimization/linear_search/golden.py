"""
黄金分割法线搜索模块。

包含一维线性搜索的黄金分割算法实现。

Author:
    LONG QIANG (Created on Thu Mar 24 19:05:38 2022)

Revisions:
    2026-04-06 [ReFantasy]: 为 Golden 函数新增 objfun (callable) 参数以传入目标函数，方便外部调用和复用
"""


## 黄金分割法
def golden(objfun: callable, a: float, b: float, epsilon: float) -> tuple[float, float, int]:
    """
    [精确线性搜索] 使用黄金分割法（0.618法）求解一元函数在给定区间内的极小值点。

    Args:
    ----------
        objfun (callable): 目标函数。
        a (float): 初始搜索区间的左端点。
        b (float): 初始搜索区间的右端点。
        epsilon (float): 容许误差，用于控制算法精度。

    Returns:
    ----------
        tuple[float, float, int]: 包含近似极小值点 xstar、对应的函数极小值 fstar 以及迭代次数 k 的元组。
    """
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
        # print(k,ak,bk,bk-ak)
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
    jax.config.update("jax_enable_x64", True)

    @jax.jit
    def objfun(x):
        y = 2 * x**2 - 4 * x - 1
        return y

    # 输入
    a = -4.0  # 初始区间左端点
    b = 4.0  # 初始区间右端点
    epsilon = 0.1  # 容忍精度

    # 黄金分割法
    xstar, fstar, k = golden(objfun, a, b, epsilon)
    print("Golden Method: ", xstar, fstar, k)
