import jax
from functools import singledispatch
from jax import grad, jacrev
import jax.numpy as jnp

def newton(func, x0, tol):
    """
    Newton's method for single nonlinear equation
    """
    grad_f = jax.jit(grad(func))
    xk_c = x0
    xk_n = x0 - func(xk_c) / grad_f(xk_c)
    while abs(xk_n - xk_c) > tol:
        xk_c = xk_n
        xk_n = xk_c - func(xk_c) / grad_f(xk_c)
    return xk_n

# def newton_mp(func, x0, x1,  tol):
#     """
#     Multi-point Newton's method
#     """
#     def gx(func,x0, x1):
#         return x1 - (x1-x0)/(func(x1)-func(x0)) * func(x1)
#     xk_p = x0
#     xk_c = x1
#     xk_n = gx(func, xk_p, xk_c)
#     while abs(xk_n - xk_c) > tol:
#         xk_p = xk_c
#         xk_c = xk_n
#         xk_n = gx(func, xk_p, xk_c)
#     return xk_n

def newton_mp(func, x0, x1,  tol):
    """
    Multi-point Newton's method
    """
    def jax_func(x):
        return jnp.array(func(x))
    
    def gx(f,x0, x1):
        return x1 - (x1-x0)/(f(x1)-f(x0)) * f(x1)
    x0 = jnp.array(x0)
    x1 = jnp.array(x1)
    xk_p = x0
    xk_c = x1
    xk_n = gx(jax_func, xk_p, xk_c)
    while abs(xk_n - xk_c) > tol:
        xk_p = xk_c
        xk_c = xk_n
        xk_n = gx(jax_func, xk_p, xk_c)
    return xk_n

def newtons(func, x0, tol):
    """
    Newton's method for systems of nonlinear equations
    """
    def jax_func(x):
        return jnp.array(func(x))
    x0 = jnp.array(x0)

    grad_f = jacrev(jax_func)
    xk_c = x0

    H = grad_f(xk_c)
    H_inv = jnp.linalg.inv(H)
    xk_n = x0 - H_inv@ jax_func(xk_c) 

    while jnp.linalg.norm(xk_n - xk_c) > tol:
        xk_c = xk_n
        H = grad_f(xk_c)
        H_inv = jnp.linalg.inv(H)
        xk_n = xk_c - H_inv@jax_func(xk_c) 
    return xk_n
