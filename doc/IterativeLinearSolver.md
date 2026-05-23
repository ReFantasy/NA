# Iterative Linear Solver
An iterative solver for $Ax=b$ has the form:

$$
x^{k+1} = x^{k} + \alpha M^{-1}(b-Ax^k) 
$$

why does it work?

$$
\begin{aligned}
b-Ax^{k+1} &= b - A (x^{k} + \alpha M^{-1}(b-Ax^k) )\\
           &= b-Ax^k-\alpha AM^{-1}(b-Ax^k)\\
           &= (I-\alpha AM^{-1})(b-Ax^k)\\
           &= (I-\alpha AM^{-1})^{k+1}(b-Ax^0)
\end{aligned}
$$

so, if $\rho(I-\alpha AM^{-1})<1$, $b-Ax^{k+1} \rightarrow 0$, $\rho(\cdot)$ is spectral radius (the largest absolute value of the eigenvalues).

## Algorithm
let $M \approx A, \alpha = 1$, then

$$
x^{k+1} = x^k + M^{-1}(b-Ax^k)
$$

let $e^k = M^{-1}(b-Ax^k)$, we have the algorithm:

```plain
for i = 0:
    Compute r = b-Ax
    Solver the error equation Me=r (solve Ae = r approximately)
    Update x = x + e
```
