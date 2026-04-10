import NumericalOptimization as optimizer
import jax
import jax.numpy as jnp
from NumericalOptimization.utils import functions


def main():
    print("Hello from na!")


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)

    objfun = functions.boha1

    x0 = jnp.array([80.0, -30.0])

    xstar, fstar, k = optimizer.gradient_methods.newton_goldstein(
        objfun, x0, epsilon=0.001, line_search_name="golden"
    )

    print(xstar, fstar, k)



# import numpy as np
# import jax
# from NumericalOptimization.utils import functions

# ## 原问题目标函数
# # def objfun(x):
# #     y = x[0]**4+x[0]*x[1]+(1+x[1])**2
# #     return y
# objfun = functions.boha1

# ## 原问题目标函数的梯度函数
# # def gradfun(x):
# #     y =np.array([4.*x[0]**3+x[1],x[0]+2.*(1.+x[1])])
# #     return y
# gradfun = jax.grad(objfun)

# ## 原问题目标函数的海森矩阵函数
# # def hessianfun(x):
# #     y = np.array([[12.*x[0]**2,1.],[1.,2.]])
# #     return y
# hessianfun = jax.hessian(objfun)

# ## 一维搜索子问题目标函数
# def lineobjfun(xk,dk,alpha):
#     y = objfun(xk+alpha*dk)
#     return y

# ## 黄金分割法
# def Golden(xk,dk,ak,bk,epsilon):
#     lambdak = ak+0.382*(bk-ak)
#     muk = ak+0.618*(bk-ak)
#     flambdak = lineobjfun(xk,dk,lambdak)
#     fmuk = lineobjfun(xk,dk,muk)
#     while True:
#         if bk-ak <= epsilon: 
#             xstar = (ak+bk)/2
#             return xstar
#         else:
#             if flambdak > fmuk:
#                 ak = lambdak
#                 lambdak = muk
#                 flambdak = fmuk
#                 muk = ak+0.618*(bk-ak)
#                 fmuk = lineobjfun(xk,dk,muk)
#             else:
#                 bk = muk
#                 muk = lambdak
#                 fmuk = flambdak
#                 lambdak = ak+0.382*(bk-ak)
#                 flambdak = lineobjfun(xk,dk,lambdak)

# ## Goldstein-Price修正牛顿法
# def Goldstein_Price_Newton(x0,epsilon):
#     eta = 0.3
#     xk = x0.copy()
#     k = 0
#     while True:
#         k += 1
#         gk = gradfun(xk)
#         if np.linalg.norm(gk) <= epsilon:
#             xstar = xk
#             fstar = objfun(xk)
#             return xstar,fstar,k
#         else:
#             Gk = hessianfun(xk)             # 海森矩阵
#             Gk_inv = np.linalg.inv(Gk)      # 海森矩阵的逆
#             dkN = -np.dot(Gk_inv,gk)        # 牛顿方向
            
#             coss = np.dot(dkN,-gk)/(np.linalg.norm(dkN)*np.linalg.norm(-gk))
#             if  coss >= eta:
#                 dk = dkN                    # 使用牛顿方向搜索
#             else:
#                 dk = -gk                    # 使用最速下降方向搜索
            
#             lambdak = Golden(xk,dk,0.,3.,0.001) # 精确一维搜索
#             xk += lambdak*dk

# ## 主程序
# if __name__ == '__main__':
#     epsilon = 0.001
    
#     ## 第1组
#     x0 = np.array([80.,-30.])
#     xstar,fstar,k = Goldstein_Price_Newton(x0,epsilon)
#     print(xstar, fstar, k)
#     # [7.76111772e-07 2.66237900e-07] 1.100841640067074e-11 10
    