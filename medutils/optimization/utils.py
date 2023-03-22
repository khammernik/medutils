"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

import numpy as np

def normest(A, n=100, tol=1e-6):
    """ Estimate 2-norm of a given matrix A
    :param A: input matrix (np.ndarray)
    :param n: number of iterations
    :param tol: relative tolerance
    :return: estimated matrix norm
    """
    A = A.astype(np.float32)
    x = np.sum(np.abs(A), 0)
    e = np.linalg.norm(x)

    x /= e
    e0 = 0
    for i in range(n):
        if np.abs(e-e0) <= tol*e:
            break
        e0 = e
        Ax = np.dot(A, x)
        x = np.dot(A.T, Ax)
        normx = np.linalg.norm(x)
        e = normx / np.linalg.norm(Ax)
        x /= normx
    if i == n-1:
        print(Warning('[WARNING] normest did not converge for n=%d iterations' % n))
    return e