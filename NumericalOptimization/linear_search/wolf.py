"""
Armijo-Goldstein步长准则

包含Armijo-Goldstein步长准则算法实现。

Author:
    LONG QIANG (Created on Thu Mar 24 20:33:23 2022)

Revisions:
    2026-04-07 [ReFantasy]: 修改逻辑错误：当搜索区间都在第二条直线下方时，原代码将进入死循环，无法正确更新 bl 的值。
                            删除原有代码：
                                        if bl < 100.0:
                                            alphal = (al + bl) / 2.0
                                        else:
                                            alphal = t * alphal
                            新增(替换)代码：
                                        if np.dot(gradfun(xk + alphal * dk), dk) < sigma * np.dot(gradfun(xk), dk):
                                            bl = bl * t
                                        alphal = (al + bl) / 2.0
"""

import sys
import jax
import jax.numpy as jnp


## Wolf-Powell步长准则
def wolf_powell(objfun, xk, dk, a0=0.0, b0=sys.float_info.max, alpha0=1.0, rho=0.3, t=1.1, sigma=0.5, gradfun=None):
    if gradfun == None:
        gradfun = jax.grad(objfun)

    def phi(xk, dk, alpha):
        y = objfun(xk + alpha * dk)
        return y

    assert 0 < rho < 0.5, "rho must be in (0, 1)"
    assert t > 1.0, "t must be greater than 1"
    assert rho < sigma < 1, "sigma must be in (rho, 1)"
    assert a0 < alpha0 < b0, "alpha0 must be in (a0, b0)"

    al, bl, alphal = a0, b0, alpha0
    l = 0

    while True:
        if phi(xk, dk, alphal) <= objfun(xk) + rho * jnp.dot(gradfun(xk), dk) * alphal:
            if jnp.dot(gradfun(xk + alphal * dk), dk) >= sigma * jnp.dot(gradfun(xk), dk):
                error = objfun(xk) - objfun(xk + alphal * dk)  # 下降量
                return alphal, l, error
            else:
                al = alphal

                if jnp.dot(gradfun(xk + alphal * dk), dk) < sigma * jnp.dot(gradfun(xk), dk):
                    bl = bl * t
                alphal = (al + bl) / 2.0

                l += 1
                continue
        else:
            bl = alphal
            alphal = (al + bl) / 2.0
            l += 1
            continue


## 主函数
if __name__ == "__main__":
    import jax

    jax.config.update("jax_enable_x64", True)

    ## 原问题目标函数
    def objfun(x):
        y = (x[0] - 1.0) ** 2.0 + (x[1] + 1.0) ** 2.0
        return y

    ##原问题目标函数的梯度函数
    # def gradfun(x):
    #     y = np.array([2 * (x[0] - 1), 2 * (x[1] + 1)])
    #     return y
    gradfun = jax.grad(objfun)

    # 输入
    xk = jnp.array([0.0, 0.0])  # 原目标函数当前跌点
    dk = -gradfun(xk)  # 当前搜索方向
    a0, b0, alpha0 = 0.0, 20.0, 10.0  # 初始区间和初始试探点
    # a0, b0, alpha0 = 0.0, 0.00001, 0.0000001  # 初始区间和初始试探点

    # Wolf-Powell步长准则
    alphal, l, error = wolf_powell(objfun, xk, dk, a0, b0, alpha0)
    print("Wolf-Powell步长准则: ", alphal, l, error)

    # alphal, l, error = Wolf_Powell(objfun, xk, dk)
    # print("Wolf-Powell步长准则: ", alphal, l, error)
