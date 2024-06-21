def riemann(f, a, b, n):
    """
    Compute the Riemann sum of a function over an interval.

    Parameters
    ----------
    f : function
        The function to integrate.
    a : float
        The lower bound of the interval.
    b : float
        The upper bound of the interval.
    n : int
        The number of subintervals to use.

    Returns
    -------
    area : float
        The estimated area under the curve.

    """
    dx = (b - a) / n
    area = 0
    for i in range(n):
        area += f(a + i * dx) * dx

    return area

def f(x):
    return x**2 + 1

def main():
    area = riemann(f, 0, 1, 1000)
    print(area)

if __name__ == "__main__":
    main()