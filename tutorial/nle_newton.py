import NumercialAnalysis as na
import numpy as np
from jax import grad, hessian, jacfwd, jacrev
import jax.numpy as jnp

def ff(x):
    return 2*x*x*x - 5*x -1

def ff_vec(x):
    x1 = x[0]
    x2 = x[1]
    
    v1 = 4*x1**2 + 3*x2**2 -1
    v2 = x1**3 -8*x2**3 -1
    # return np.array([v1, v2])
    # return jnp.array([v1, v2])
    return [v1, v2]

if __name__ == "__main__":
    print("Using Newton's method solve single nonlinear equation")
    x = na.NLE.newton(ff, x0=1.5, tol=1e-4)
    print("The root is:",x)
    print("absolute error:", 0.0 - ff(x))

    print("\nUsing Multi-point Newton's method solve single nonlinear equation")
    x = na.NLE.newton_mp(ff, x0=1.5, x1 = 1.6, tol=1e-4)
    print("The root is:",x)
    print("absolute error:", 0.0 - ff(x))

    print("\nUsing Newton's method solve systems of nonlinear equations")
    x = [0.1,0.1]
    jnp_xn = na.NLE.newtons(ff_vec, x0=x, tol=1e-4)
    print("The root is:",jnp_xn)
    print("absolute error:", np.array(ff_vec(jnp_xn)))
