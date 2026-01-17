import NumercialAnalysis as na

def ff(x):
    return x*x*x - x -1

if __name__ == "__main__":
    # Using Aitken's method
    # iter function is iter_func: f(x) = x*x*x -1
    x = na.NLE.aitken(lambda x: x*x*x -1, x0=1.5, tol=1e-4)
    print("The root is:",x)

