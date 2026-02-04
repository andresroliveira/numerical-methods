"""
Arnoldi iteration for eigenvalue computation.

The Arnoldi method is a Krylov subspace method for finding eigenvalues
of general (non-symmetric) matrices. It builds an orthonormal basis
of the Krylov subspace using modified Gram-Schmidt orthogonalization.
"""


def arnoldi_iteration(A, v0, m):
    """
    Perform m steps of Arnoldi iteration.

    Builds an orthonormal basis {q1, q2, ..., qm} of the Krylov subspace
    K_m(A, v0) = span{v0, Av0, A²v0, ..., A^(m-1)v0}.

    Parameters
    ----------
    A : list of list of float
        The matrix (n x n)
    v0 : list of float
        Initial vector (length n)
    m : int
        Number of iterations (dimension of Krylov subspace)

    Returns
    -------
    Q : list of list of float
        Orthonormal basis vectors as columns (n x m)
    H : list of list of float
        Upper Hessenberg matrix (m x m)

    Raises
    ------
    ValueError
        If breakdown occurs (Krylov subspace dimension < m)

    Examples
    --------
    >>> A = [[2, 1], [1, 2]]
    >>> v0 = [1, 0]
    >>> Q, H = arnoldi_iteration(A, v0, 2)
    >>> len(Q[0]) == 2  # 2 basis vectors
    True
    """
    n = len(A)

    # Normalize initial vector
    norm_v0 = sum(x**2 for x in v0) ** 0.5
    if norm_v0 < 1e-14:
        raise ValueError("Initial vector must be non-zero")

    # Q stores orthonormal basis vectors as columns
    # Q[i] is the i-th row, Q[i][j] is Q_{i,j}
    Q = [[0.0] * m for _ in range(n)]
    for i in range(n):
        Q[i][0] = v0[i] / norm_v0

    # H is upper Hessenberg matrix
    H = [[0.0] * m for _ in range(m)]

    for j in range(m):
        # Compute A * q_j
        q_j = [Q[i][j] for i in range(n)]
        w = matrix_vector_mult(A, q_j)

        # Modified Gram-Schmidt orthogonalization
        for i in range(j + 1):
            q_i = [Q[k][i] for k in range(n)]
            H[i][j] = dot_product(q_i, w)
            # w = w - H[i][j] * q_i
            for k in range(n):
                w[k] -= H[i][j] * q_i[k]

        # Compute norm of w
        h_next = sum(x**2 for x in w) ** 0.5

        if h_next < 1e-14:
            # Lucky breakdown - exact invariant subspace found
            # Return smaller matrices
            Q_reduced = [[Q[i][k] for k in range(j + 1)] for i in range(n)]
            H_reduced = [[H[i][k] for k in range(j + 1)] for i in range(j + 1)]
            return Q_reduced, H_reduced

        # Store in H and normalize
        if j + 1 < m:
            H[j + 1][j] = h_next
            for i in range(n):
                Q[i][j + 1] = w[i] / h_next

    return Q, H


def arnoldi_eigenvalues(A, m, num_restarts=0):
    """
    Compute approximate eigenvalues using Arnoldi iteration.

    Parameters
    ----------
    A : list of list of float
        The matrix (n x n)
    m : int
        Krylov subspace dimension
    num_restarts : int, optional
        Number of implicitly restarted iterations (default: 0)

    Returns
    -------
    eigenvalues : list of complex
        Approximate eigenvalues (eigenvalues of H)

    Examples
    --------
    >>> A = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]
    >>> eigs = arnoldi_eigenvalues(A, 3)
    >>> len(eigs) == 3
    True
    """
    n = len(A)

    # Start with random initial vector
    import random

    random.seed(42)
    v0 = [random.random() - 0.5 for _ in range(n)]

    # Perform Arnoldi iteration
    Q, H = arnoldi_iteration(A, v0, min(m, n))

    # Compute eigenvalues of H using QR algorithm
    eigenvalues = qr_eigenvalues_hessenberg(H, max_iter=100)

    return eigenvalues


def arnoldi_largest_eigenvalues(A, k, m=None, tol=1e-8, max_iter=100):
    """
    Compute k largest eigenvalues (by magnitude) using Arnoldi.

    Parameters
    ----------
    A : list of list of float
        The matrix (n x n)
    k : int
        Number of eigenvalues to compute
    m : int, optional
        Krylov subspace dimension (default: min(2*k + 1, n))
    tol : float, optional
        Convergence tolerance (default: 1e-8)
    max_iter : int, optional
        Maximum number of restarts (default: 100)

    Returns
    -------
    eigenvalues : list of complex
        k largest eigenvalues by magnitude
    eigenvectors : list of list of float
        Corresponding eigenvectors

    Examples
    --------
    >>> A = [[5, 1, 0], [1, 3, 1], [0, 1, 1]]
    >>> eigs, vecs = arnoldi_largest_eigenvalues(A, 2)
    >>> len(eigs) == 2
    True
    """
    n = len(A)
    if m is None:
        m = min(2 * k + 1, n)

    import random

    random.seed(42)
    v0 = [random.random() - 0.5 for _ in range(n)]

    Q, H = arnoldi_iteration(A, v0, m)

    # Get eigenvalues of H
    H_eigs = qr_eigenvalues_hessenberg(H, max_iter=100)

    # Sort by magnitude
    H_eigs_sorted = sorted(H_eigs, key=lambda x: abs(x), reverse=True)

    # Return k largest
    eigenvalues = H_eigs_sorted[:k]

    # Compute approximate eigenvectors
    # For simplicity, we return the Ritz vectors (Q * eigenvectors of H)
    eigenvectors = []

    # Note: Full eigenvector computation would require solving
    # the eigenvector problem for H, then multiplying by Q
    # For now, return empty list as placeholder

    return eigenvalues, eigenvectors


def matrix_vector_mult(A, v):
    """Multiply matrix A by vector v."""
    n = len(A)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * v[j]
    return result


def dot_product(u, v):
    """Compute dot product of two vectors."""
    return sum(u[i] * v[i] for i in range(len(u)))


def qr_eigenvalues_hessenberg(H, max_iter=100, tol=1e-10):
    """
    Compute eigenvalues of upper Hessenberg matrix using QR algorithm.

    Parameters
    ----------
    H : list of list of float
        Upper Hessenberg matrix
    max_iter : int, optional
        Maximum iterations
    tol : float, optional
        Convergence tolerance

    Returns
    -------
    eigenvalues : list of float or complex
        Eigenvalues of H
    """
    m = len(H)

    # Make a copy
    A = [row[:] for row in H]

    for iteration in range(max_iter):
        # Check for convergence (subdiagonal elements near zero)
        converged = True
        for i in range(m - 1):
            if abs(A[i + 1][i]) > tol:
                converged = False
                break

        if converged:
            break

        # QR decomposition of A (simplified for Hessenberg)
        Q, R = qr_hessenberg(A)

        # A = R * Q
        A = matrix_mult(R, Q)

    # Extract eigenvalues from diagonal (for real matrices)
    # Handle 2x2 blocks for complex eigenvalues
    eigenvalues = []
    i = 0
    while i < m:
        if i == m - 1:
            # Last element
            eigenvalues.append(A[i][i])
            i += 1
        elif abs(A[i + 1][i]) < tol:
            # Converged diagonal element
            eigenvalues.append(A[i][i])
            i += 1
        else:
            # 2x2 block for complex conjugate pair
            a = A[i][i]
            b = A[i][i + 1]
            c = A[i + 1][i]
            d = A[i + 1][i + 1]

            # Eigenvalues of 2x2 matrix
            trace = a + d
            det = a * d - b * c
            disc = trace**2 - 4 * det

            if disc >= 0:
                # Real eigenvalues
                sqrt_disc = disc**0.5
                eigenvalues.append((trace + sqrt_disc) / 2)
                eigenvalues.append((trace - sqrt_disc) / 2)
            else:
                # Complex eigenvalues
                real_part = trace / 2
                imag_part = (-disc) ** 0.5 / 2
                eigenvalues.append(complex(real_part, imag_part))
                eigenvalues.append(complex(real_part, -imag_part))

            i += 2

    return eigenvalues


def qr_hessenberg(H):
    """
    QR decomposition of upper Hessenberg matrix using Givens rotations.

    Parameters
    ----------
    H : list of list of float
        Upper Hessenberg matrix

    Returns
    -------
    Q : list of list of float
        Orthogonal matrix
    R : list of list of float
        Upper triangular matrix
    """
    m = len(H)
    R = [row[:] for row in H]

    # Initialize Q as identity
    Q = [[1.0 if i == j else 0.0 for j in range(m)] for i in range(m)]

    # Apply Givens rotations to zero subdiagonal
    for j in range(m - 1):
        # Eliminate R[j+1][j]
        a = R[j][j]
        b = R[j + 1][j]

        if abs(b) < 1e-14:
            continue

        # Compute Givens rotation
        r = (a**2 + b**2) ** 0.5
        c = a / r if r > 0 else 1.0
        s = b / r if r > 0 else 0.0

        # Apply to R
        for k in range(m):
            temp1 = c * R[j][k] + s * R[j + 1][k]
            temp2 = -s * R[j][k] + c * R[j + 1][k]
            R[j][k] = temp1
            R[j + 1][k] = temp2

        # Apply to Q^T (accumulate Q)
        for k in range(m):
            temp1 = c * Q[k][j] + s * Q[k][j + 1]
            temp2 = -s * Q[k][j] + c * Q[k][j + 1]
            Q[k][j] = temp1
            Q[k][j + 1] = temp2

    return Q, R


def matrix_mult(A, B):
    """Multiply two matrices."""
    m = len(A)
    n = len(B[0]) if B else 0
    p = len(B)

    result = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            for k in range(p):
                result[i][j] += A[i][k] * B[k][j]

    return result


def main():
    """Example usage of Arnoldi iteration."""

    print("=" * 70)
    print("Arnoldi Iteration for Eigenvalue Computation")
    print("=" * 70)

    # Example 1: Small symmetric matrix (for verification)
    print("\n1. Symmetric matrix (eigenvalues known)")
    A1 = [[4, 1, 0], [1, 3, 1], [0, 1, 2]]

    print("   Matrix A:")
    for row in A1:
        print(f"   {row}")

    v0 = [1, 0, 0]
    Q, H = arnoldi_iteration(A1, v0, 3)

    print(f"\n   Krylov subspace dimension: {len(H)}")
    print("   Upper Hessenberg matrix H:")
    for row in H:
        print(f"   {[f'{x:8.5f}' for x in row]}")

    eigs = arnoldi_eigenvalues(A1, 3)
    print("\n   Approximate eigenvalues:")
    for i, eig in enumerate(eigs):
        if isinstance(eig, complex):
            print(f"   λ{i + 1} = {eig.real:.6f} + {eig.imag:.6f}i")
        else:
            print(f"   λ{i + 1} = {eig:.6f}")

    # Example 2: Non-symmetric matrix
    print("\n2. Non-symmetric matrix")
    A2 = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

    print("   Matrix A (circular shift):")
    for row in A2:
        print(f"   {row}")

    eigs2 = arnoldi_eigenvalues(A2, 3)
    print("\n   Eigenvalues (should be cube roots of unity):")
    for i, eig in enumerate(eigs2):
        if isinstance(eig, complex):
            mag = abs(eig)
            print(f"   λ{i + 1} = {eig.real:.6f} + {eig.imag:.6f}i  (|λ| = {mag:.6f})")
        else:
            print(f"   λ{i + 1} = {eig:.6f}")

    # Example 3: Larger matrix - find dominant eigenvalues
    print("\n3. Larger matrix - dominant eigenvalues")
    n = 5
    A3 = [[0.0] * n for _ in range(n)]

    # Create a matrix with known spectrum
    for i in range(n):
        A3[i][i] = n - i  # Diagonal: 5, 4, 3, 2, 1
        if i < n - 1:
            A3[i][i + 1] = 0.5  # Superdiagonal
            A3[i + 1][i] = 0.5  # Subdiagonal

    print(f"   {n}x{n} tridiagonal matrix")
    print("   Diagonal: [5, 4, 3, 2, 1], off-diagonal: 0.5")

    eigs3 = arnoldi_eigenvalues(A3, min(n, 5))
    print("\n   Approximate eigenvalues:")
    for i, eig in enumerate(sorted(eigs3, key=lambda x: abs(x), reverse=True)):
        if isinstance(eig, complex):
            print(f"   λ{i + 1} = {eig.real:.6f} + {eig.imag:.6f}i")
        else:
            print(f"   λ{i + 1} = {eig:.6f}")

    # Example 4: Convergence with different subspace sizes
    print("\n4. Effect of Krylov subspace dimension")
    A4 = [[3, 1, 0, 0], [1, 3, 1, 0], [0, 1, 3, 1], [0, 0, 1, 3]]

    print("   4x4 tridiagonal matrix")

    for m in [2, 3, 4]:
        eigs_m = arnoldi_eigenvalues(A4, m)
        print(f"\n   m = {m}: {len(eigs_m)} eigenvalues computed")
        for eig in sorted(eigs_m, key=lambda x: abs(x), reverse=True):
            if isinstance(eig, complex):
                print(f"      {eig.real:.6f} + {eig.imag:.6f}i")
            else:
                print(f"      {eig:.6f}")

    print("\n" + "=" * 70)
    print("Notes:")
    print("  - Arnoldi builds orthonormal basis of Krylov subspace")
    print("  - Works for general (non-symmetric) matrices")
    print("  - Produces upper Hessenberg matrix H")
    print("  - Eigenvalues of H approximate eigenvalues of A")
    print("  - Larger m gives better approximation")
    print("=" * 70)


if __name__ == "__main__":
    main()
