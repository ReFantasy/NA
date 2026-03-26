import NumercialAnalysis as na
import jax.numpy as jnp


if __name__ == "__main__":
    A = jnp.array([[6, -2, 1], 
                   [-2, 5, 3], 
                   [1, 3, -4]])
    
    Q, lam = na.Eigen.jacobi(A, tol=1e-13)
    print("Eigenvalues:\n", lam)
    print("Eigenvectors (columns of Q):\n", Q)

    print("Check error")
    for i in range(3):
        print(Q[:,i] * lam[i] - A @ Q[:,i])
  