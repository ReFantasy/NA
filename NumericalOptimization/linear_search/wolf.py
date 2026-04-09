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
