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

#import numpy as np
import sys
import jax 
import jax.numpy as jnp


## 原问题目标函数
# def objfun(x):
#     y = (x[0] - 1) ** 2 + (x[1] + 1) ** 2
#     return y


##原问题目标函数的梯度函数
def gradfun(x):
    y = jnp.array([2 * (x[0] - 1), 2 * (x[1] + 1)])
    return y


## 一维搜索目标函数
# def phi(xk, dk, alpha):
#     y = objfun(xk + alpha * dk)
#     return y


## Armijo-Goldstein步长准则
def Armijo_Goldstein(objfun, xk, dk, a0, b0=sys.float_info.max, alpha0=1.0):
    def phi(xk, dk, alpha):
        y = objfun(xk + alpha * dk)
        return y

    rho, t = 0.3, 1.1
    al, bl, alphal = a0, b0, alpha0
    l = 0

    while True:
        if phi(xk, dk, alphal) <= objfun(xk) + rho * jnp.dot(gradfun(xk), dk) * alphal:
            if phi(xk, dk, alphal) >= objfun(xk) + (1 - rho) * jnp.dot(gradfun(xk), dk) * alphal:
                error = objfun(xk) - objfun(xk + alphal * dk)  # 下降量
                return alphal, l, error
            else:
                al = alphal
                ########################################
                # delete
                ########################################
                # if bl < 100.0:
                #     alphal = (al + bl) / 2.0
                # else:
                #     alphal = t * alphal
                ########################################

                ########################################
                # new
                ########################################
                if phi(xk, dk, alphal) < objfun(xk) + (1 - rho) * jnp.dot(gradfun(xk), dk) * alphal:
                    bl = bl * t
                alphal = (al + bl) / 2.0
                ########################################

                l += 1
                continue
        else:
            bl = alphal
            alphal = (al + bl) / 2.0
            l += 1
            continue


## 主函数
if __name__ == "__main__":

    def objfun(x):
        y = (x[0] - 1) ** 2 + (x[1] + 1) ** 2
        return y
    
    # 输入
    xk = jnp.array([0, 0])  # 原目标函数当前跌点
    dk = -gradfun(xk)  # 当前搜索方向
  
    a0,b0,alpha0 = 0.,20.,10.   # 初始区间和初始试探点 Armijo-Goldstein步长准则:  (0.625, 4, np.float64(1.875))
    #a0, b0, alpha0 = 0.0, 0.00001, 0.0000001  # 初始区间和初始试探点

    # Armijo-Goldstein步长准则
    print("Armijo-Goldstein步长准则: ", Armijo_Goldstein(objfun, xk, dk, a0, b0, alpha0))

    # print( 10< float('inf') )
    # print( sys.float_info.max )

