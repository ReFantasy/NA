import numpy as np
import sys

## 原问题目标函数
def objfun(x):
    y = (x[0] - 1) ** 2 + (x[1] + 1) ** 2
    return y


##原问题目标函数的梯度函数
def grad(x):
    y = np.array([2 * (x[0] - 1), 2 * (x[1] + 1)])
    return y


## 一维搜索目标函数
def phi(xk, dk, alpha):
    y = objfun(xk + alpha * dk)
    return y


## Armijo-Goldstein步长准则
def Armijo_Goldstein(xk, dk, a0, b0 = sys.float_info.max, alpha0 = 1.0):
    rho, t = 0.3, 1.1
    al, bl, alphal = a0, b0, alpha0
    l = 0

    while True:
        if phi(xk, dk, alphal) <= objfun(xk) + rho * np.dot(grad(xk), dk) * alphal:
            if phi(xk, dk, alphal) >= objfun(xk) + (1 - rho) * np.dot(grad(xk), dk) * alphal:
                error = objfun(xk) - objfun(xk + alphal * dk)  # 下降量
                return alphal, l, error
            else:
                al = alphal
                # if bl < 100.0:
                #     alphal = (al + bl) / 2.0
                # else:
                #     alphal = t * alphal
                
                if phi(xk, dk, alphal)< objfun(xk) + (1 - rho) * np.dot(grad(xk), dk) * alphal:
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
if __name__ == '__main__':
    # 输入
    xk = np.array([0,0])        # 原目标函数当前跌点
    dk = -grad(xk)              # 当前搜索方向
    #a0,b0,alpha0 = 0.,20.,10.   # 初始区间和初始试探点 Armijo-Goldstein步长准则:  (0.625, 4, np.float64(1.875))
    a0,b0,alpha0 = 0., 0.00001, 0.0000001   # 初始区间和初始试探点
    
    # Armijo-Goldstein步长准则
    print('Armijo-Goldstein步长准则: ', Armijo_Goldstein(xk,dk,a0,b0,alpha0))

    # print( 10< float('inf') )
    # print( sys.float_info.max )