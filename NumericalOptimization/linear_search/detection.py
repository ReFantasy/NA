import jax.numpy as jnp


def detection(objfun, xk, dk, alpha0, alpha_min=0.001, beta=0.9, gamma=1.1):
    """
    探测搜索算法，用于在数值优化中(直接方法)寻找合适的步长。

    参数:
    ---
        objfun (callable): 目标函数，接受一个向量输入并返回一个标量输出。
        xk (jax.numpy.ndarray): 当前迭代点。
        dk (jax.numpy.ndarray): 搜索方向。
        alpha0 (float): 初始步长。
        alpha_min (float, optional): 最小步长，默认为 0.001。
        beta (float, optional): 步长缩短因子，默认为 0.9。
        gamma (float, optional): 步长伸长因子，默认为 1.1。
    返回:
    ---
        float: 计算得到的最优步长。
    """

    def phi(xk, dk, ss):
        return objfun(xk + ss * dk)

    alpha = alpha0  # 初始步长

    # 主循环
    while True:
        # dk为下降方向
        if phi(xk, dk, alpha) < phi(xk, dk, 0):
            # print("Descent direction: dk={}".format(dk))
            # print("Initial step size: lambdak={}".format(lambdak))
            while True:
                lambda_prime = gamma * alpha  # 伸长步长
                if phi(xk, dk, lambda_prime) < phi(xk, dk, alpha):
                    alpha = lambda_prime  # 确认新步长
                    # print("Increase step size: lambdak={}".format(lambdak))
                else:
                    lambda_star = alpha  # 最优步长
                    # print("Optimal step size: lambda_star={}".format(lambda_star))
                    return lambda_star
        # -dk为下降方向
        elif phi(xk, -dk, alpha) < phi(xk, dk, 0):
            # print("Descent direction: -dk={}".format(-dk))
            # print("Initial step size: lambdak={}".format(lambdak))
            while True:
                lambda_prime = gamma * alpha  # 缩短步长
                if phi(xk, -dk, lambda_prime) < phi(xk, -dk, alpha):
                    alpha = lambda_prime  # 确认新步长
                    # print("Increase step size: lambdak={}".format(lambdak))
                else:
                    lambda_star = alpha  # 最优步长
                    # print("Optimal step size: lambda_star={}".format(lambda_star))
                    return -lambda_star
        # 缩短初始试探步长
        else:
            alpha = beta * alpha  # 缩短步长
            # print("Decrease step size: lambdak={}".format(lambdak))
            if alpha < alpha_min:
                lambda_star = 0.0  # 超过阈值，无有效步长
                # print("Nondescent direction: lambda_star={}".format(lambda_star))
                return lambda_star
            else:
                continue


## 主程序
if __name__ == "__main__":
    import jax
    from NumericalOptimization.utils.draw import draw1d

    @jax.jit
    def objfun(x):
        y = (x[0] - 3.0) ** 2 + 2.0 * (x[1] + 2) ** 2
        return y

    lambda0 = 1.0  # 初始步长

    ## 第1组
    xk = jnp.array([1.0, 0.0])  # 当前迭代点
    dk = jnp.array([0.0, 1.0])  # 搜索方向

    print("xk={}, dk={}, initial alpha={}".format(xk, dk, lambda0))
    lambda_star = detection(objfun, xk, dk, lambda0)
    print("alpha_star={}".format(lambda_star))

    ## 第2组
    xk = jnp.array([1.0, 0.0])  # 当前迭代点
    dk = jnp.array([3.0, 1.0])  # 搜索方向
    print("\nxk={}, dk={}, initial alpha={}".format(xk, dk, lambda0))
    lambda_star = detection(objfun, xk, dk, lambda0)
    print("alpha_star={}".format(lambda_star))

    ## 第3组
    xk = jnp.array([1.0, 0.0])  # 当前迭代点
    dk = jnp.array([2.0, 1.0])  # 搜索方向
    print("\nxk={}, dk={}, initial alpha={}".format(xk, dk, lambda0))
    lambda_star = detection(objfun, xk, dk, lambda0)
    print("alpha_star={}".format(lambda_star))

    phi = lambda alpha: objfun(xk + alpha * dk)
    draw1d(phi, x_range=(-4, 4), samples=200)
