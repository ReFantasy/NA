import jax
import jax.numpy as jnp
from mpax import create_lp, r2HPDHG

def linprog(f: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray, Aeq: jnp.ndarray, beq: jnp.ndarray, lb: jnp.ndarray, ub: jnp.ndarray):
    """
    Linear programming solver
    Finds the minimum of a problem specified by

    minimize     f^T x
    subject to   A⋅x <= b
                 Aeq⋅x = beq
                 lb <= x <= ub
    f, x, b, bl, beq, lb, and ub are vectors, and A and Aeq are matrices.
    """
    n = len(f)

    if (A is None and Aeq is None) or (A.size == 0 and Aeq.size == 0):
        x = -jnp.sign(f)
        return x, f @ x
    
    if A is None or A.size == 0:
        A = jnp.zeros((1, n))
        b = jnp.zeros(1)
    if Aeq is None or Aeq.size == 0:
        Aeq = jnp.zeros((1, n))
        beq = jnp.zeros(1)
    if b is None or b.size == 0:
        b = jnp.zeros(A.shape[0])
    if beq is None or beq.size == 0:
        beq = jnp.zeros(Aeq.shape[0])
    if lb is None or lb.size == 0:
        lb = -1.0 * jnp.ones(n)
    if ub is None or ub.size == 0:
        ub = 1.0 * jnp.ones(n)
    
    # 调用 MPAX 求解线性规划
    # min  c^T x
    # s.t. Ax =  b
    #      Gx >= h
    lp = create_lp(
        c=f,
        A=Aeq,
        b=beq,
        G=-A,
        h=-b,
        l=lb,
        u=ub,
        use_sparse_matrix=False,
    )
    solver = r2HPDHG(verbose=False)
    result = solver.optimize(lp)

    x = result.primal_solution
    return x, f @ x

if __name__ == "__main__":
    f = jnp.array([-1.0, -1.0/3.0])
    A = jnp.array([[1.0, 1.0], 
                   [1.0, 0.25],
                   [1,-1],
                   [-0.25, -1],
                   [-1, -1],
                   [-1,1]], dtype=jnp.float32)
    b = jnp.array([2.0, 1.0, 2.0, 1.0, -1.0, 2.0], dtype=jnp.float32)
    Aeq = jnp.array([], dtype=jnp.float32).reshape(0, 2)
    beq = jnp.array([], dtype=jnp.float32)
    lb = jnp.array([-10.0, -10.0], dtype=jnp.float32)
    ub = jnp.array([jnp.inf, jnp.inf], dtype=jnp.float32)
    x, fx = linprog(f, A, b, Aeq, beq, lb, ub)
    print(f"Optimal solution: {x}, optimal value: {fx}")
    # Ground truth solution: x = [0.6667, 1.3333]

    Aeq = jnp.array([[1.0, 0.25]], dtype=jnp.float32)
    beq = jnp.array([0.5], dtype=jnp.float32)
    x, fx = linprog(f, A, b, Aeq, beq, lb, ub)
    print(f"Optimal solution: {x}, optimal value: {fx}")
    # Ground truth solution: x = [0, 2]

    lb = jnp.array([-1.0, -0.5], dtype=jnp.float32)
    ub = jnp.array([1.5, 1.25], dtype=jnp.float32)
    x, fx = linprog(f, A, b, Aeq, beq, lb, ub)
    print(f"Optimal solution: {x}, optimal value: {fx}")
    # Ground truth solution: x = [0.1875, 1.2500]
    
    