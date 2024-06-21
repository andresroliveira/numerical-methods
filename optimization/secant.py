def secant(f, x0, tol=1e-6, max_iter=100):
    """
    Secant method for finding a root of a function.

    Parameters
    ----------
    f : function
        The function for which to find a root.
    x0 : float
        The first initial guess for the root.
    tol : float
        The convergence tolerance.
    max_iter : int
        The maximum number of iterations.

    Returns
    -------
    x : float
        The estimated root.

    """
    x1 = x0 + 1e-4
    for _ in range(max_iter):
        x = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        if abs(x - x1) < tol:
            return x
        x0, x1 = x1, x

    return x

def f(x):
    return x**2 - 2

def main():
    root = secant(f, 1)
    print(root)

if __name__ == "__main__":
    main()