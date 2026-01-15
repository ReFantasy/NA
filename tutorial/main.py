from re import I
import NumercialAnalysis as na


def ff(x):
    return 2*x*x*x - 5*x -1


if __name__ == "__main__":
    I = [1,2]
    tol = 1e-2
    x = na.NLE.bisection(I,ff, tol=tol)
    print("The root is:",x)
    print("Check f(root):",ff(x))
