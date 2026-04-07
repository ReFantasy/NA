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
                                        if phi(xk, dk, alphal)< objfun(xk) + (1 - rho) * np.dot(grad(xk), dk) * alphal:
                                            bl = bl * t
                                        alphal = (al + bl) / 2.0
"""

import sys
import jax
import jax.numpy as jnp


## Armijo-Goldstein步长准则
def armijo_goldstein(objfun, xk, dk, a0=0.0, b0=sys.float_info.max, alpha0=1.0, rho=0.3, t=1.1, gradfun=None):
    if gradfun == None:
        gradfun = jax.grad(objfun)

    def phi(xk, dk, alpha):
        y = objfun(xk + alpha * dk)
        return y

    assert 0 < rho < 0.5, "rho must be in (0, 1)"
    assert t > 1.0, "t must be greater than 1"
    assert a0 < alpha0 < b0, "alpha0 must be in (a0, b0)"

    al, bl, alphal = a0, b0, alpha0
    l = 0

    while True:
        if phi(xk, dk, alphal) <= objfun(xk) + rho * jnp.dot(gradfun(xk), dk) * alphal:
            if phi(xk, dk, alphal) >= objfun(xk) + (1 - rho) * jnp.dot(gradfun(xk), dk) * alphal:
                error = objfun(xk) - objfun(xk + alphal * dk)  # 下降量
                return alphal, l, error
            else:
                al = alphal

                if phi(xk, dk, alphal) < objfun(xk) + (1 - rho) * jnp.dot(gradfun(xk), dk) * alphal:
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

    def objfun(x):
        y = (x[0] - 1) ** 2 + (x[1] + 1) ** 2
        return y

    gradfun = jax.grad(objfun)

    # 输入
    xk = jnp.array([0.0, 0.0])  # 原目标函数当前跌点
    dk = -gradfun(xk)  # 当前搜索方向

    a0, b0, alpha0 = 0.0, 20.0, 10.0  # 初始区间和初始试探点 Armijo-Goldstein步长准则:  (0.625, 4, np.float64(1.875))
    # a0, b0, alpha0 = 0.0, 0.00001, 0.0000001  # 初始区间和初始试探点

    alphal, l, error = armijo_goldstein(objfun, xk, dk, a0, b0, alpha0)
    print("Armijo-Goldstein步长准则: ", alphal, l, error)

    # alphal, l, error = Armijo_Goldstein(objfun, xk, dk)
    # print("Armijo-Goldstein步长准则: ", alphal, l, error)
