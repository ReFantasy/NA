import NumercialAnalysis as na
import jax.numpy as jnp
import jax
import time

def compute_time():
    # 生成一个随机的实对称矩阵
    dims = 7
    key = jax.random.PRNGKey(0)  
    A = jax.random.normal(key, (dims, dims))
    A = (A + A.T) / 2  

    start_time = time.time()
    Q, lam = na.Eigen.jacobi(A, tol=1e-13)
    end_time = time.time()
    print(f"jacobi computation time: {end_time - start_time} seconds")

    start_time = time.time()
    Q, lam = na.Eigen.passby_jacobi(A, tol=1e-13)
    end_time = time.time()
    print(f"passby_jacobi computation time: {end_time - start_time} seconds")
    
if __name__ == "__main__":
    # A = jnp.array([[6, -2, 1], 
    #                [-2, 5, 3], 
    #                [1, 3, -4]])
    # A = jnp.array([[2, -1, 0], 
    #                [-1, 2, -1], 
    #                [0, -1, 2]])
  
    
    #Q, lam = na.Eigen.jacobi(A, tol=10e-5)
    # Q, lam = na.Eigen.passby_jacobi(A, tol=10e-5)
   
    # print("Eigenvalues:\n", lam)
    # print("Eigenvectors (columns of Q):\n", Q)
    
    # print("Check error")
    # for i in range(3):
    #     print(Q[:,i] * lam[i] - A @ Q[:,i])

    compute_time()

    
  