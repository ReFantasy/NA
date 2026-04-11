import jax
import jax.numpy as jnp
from NumericalOptimization import linear_search
from NumericalOptimization.linear_search import LineSearchParams
from NumericalOptimization.utils import line_search_function

## 共轭梯度法(对一般函数带重启机制)
def conjugate_gradient(objfun, x0, epsilon, gradfun=None, hessianfun=None, line_search_params: LineSearchParams = None):
    if gradfun is None:
        gradfun = jax.jit(jax.grad(objfun))
    if hessianfun is None:
        hessianfun = jax.jit(jax.hessian(objfun))

    xk = x0
    dim = len(xk)
    k = 0
    
    while True:
        gk = gradfun(xk)            # 计算梯度
        A = hessianfun(xk)          # 计算海森矩阵
                                    # 终止条件判断
        if jnp.linalg.norm(gk) <= epsilon:
                xstar = xk
                fstar = objfun(xk)
                return xstar,fstar,k
        
        # 第一次迭代    
        dk = -gk                    # 第一次使用负梯度方向
                                    # 搜索步长

        #lambdak = -jnp.dot(gk,dk)/jnp.dot(dk,jnp.dot(A,dk))
        if line_search_params is not None:
            lambdak, _, _ = line_search_function(objfun, xk, dk, line_search_params)
        else:
            lambdak = -jnp.dot(gk,dk)/jnp.dot(dk,jnp.dot(A,dk))

        xk += lambdak*dk            # 迭代
        dk_old = dk
        k += 1
    
        # 后续迭代
        while True:
            gk = gradfun(xk)        # 计算梯度
            A = hessianfun(xk)      # 计算海森矩阵
            if jnp.linalg.norm(gk) <= epsilon:
                xstar = xk
                fstar = objfun(xk)
                return xstar,fstar,k
            else:
                                    # 计算系数beta
                betak = jnp.dot(dk_old,jnp.dot(A,gk))/jnp.dot(dk,jnp.dot(A,dk_old))
                                    # 搜索方向
                dk = -gk+betak*dk_old
                
                #lambdak = -jnp.dot(gk,dk)/jnp.dot(dk,jnp.dot(A,dk))
                if line_search_params is not None:
                    lambdak, _, _ = line_search_function(objfun, xk, dk, line_search_params)
                else:
                    lambdak = -jnp.dot(gk,dk)/jnp.dot(dk,jnp.dot(A,dk))

                xk += lambdak*dk    # 迭代
                dk_old = dk
                k += 1
                
                if jnp.mod(k,dim)==0:# 重启共轭梯度法
                    break

if __name__ == '__main__':
    from NumericalOptimization import linear_search

    import numpy

    @jax.jit
    def objfun(x):
        y = ((x[0]-3.)*(x[0]+4.))**2+((x[1]-3.)*(x[1]+4.))**2
        return y

    epsilon = 0.001

    test_groups = [
        ("第1组", jnp.array([2., 4.])),
        ("第2组", jnp.array([-3., 5.])),
        ("第3组", jnp.array([10., -10.])),
        ("第4组", jnp.array([1., -1.])),
    ]

    print(f"{'组别':<6}{'x0':<20}{'xstar':<28}{'fstar':>14}{'k':>6}")
    print("-" * 74)
    for group_name, x0 in test_groups:
        xstar, fstar, k = conjugate_gradient(objfun, x0, epsilon, line_search_params=linear_search.LineSearchParams(name=linear_search.types.golden, epsilon=0.001))
        x0_str = numpy.array2string(x0, precision=2, floatmode='fixed')
        xstar_str = numpy.array2string(xstar, precision=6, floatmode='fixed')
        print(f"{group_name:<6}{x0_str:<20}{xstar_str:<28}{fstar:>14.6e}{k:>6d}")

    # [3.00000004 3.00000004] 1.5063572340024023e-13 7
    # [-4.  3.] 8.67308323667337e-18 9
    # [ 3.00000706 -4.00000682] 4.721974485918222e-09 11
    # [-0.5 -0.5] 300.125 7
