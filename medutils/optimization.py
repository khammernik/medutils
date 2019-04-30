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

class CgSenseReconstruction(object):
    """ CG Reconstruction using the algorithm in [1].

    [1] Pruessmann, K. P.; Weiger, M.; Boernert, P. and Boesiger, P.
        Advances in sensitivity encoding with arbitrary k-space trajectories.
        Magn Reson Med 46: 638-651 (2001)
    """
    def __init__(self, op, alpha=0, tol=1e-6, max_iter=50):
        """ Initialization
        :param op: operator class containing a forward and adjoint method
        :param alpha: Tikohonov regularization parameter
        :param tol: relative tolerance
        :param max_iter: maximum number of iterations
        """
        self._alpha = alpha
        self._tol = tol
        self._max_iter = max_iter
        self.op = op

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        self._alpha = value

    @property
    def tol(self):
        return self._tol

    @tol.setter
    def tol(self, value):
        self._tol = value

    @property
    def max_iter(self):
        return self._max_iter

    @max_iter.setter
    def max_iter(self, value):
        self._max_iter = value
        
    def complexDot(self, u, v):
        """ Compute complex dot product
        :param u: np.array
        :param v: np.array
        :return: complex dot product of u and v
        """
        return np.dot(np.conjugate(u.flatten()) , v.flatten())
    
    def normSquared(self, u):
        """ Compute squared norm
        :param u: np.array
        :return: squared norm of u
        """
        return np.real(np.dot(np.conjugate(u.flatten()) , u.flatten()))

    def __systemMatrix__(self, x):
        """ Compute result on system matrix A^H * A + alpha * I
        :param x: np.array
        :return: result for system matrix applied on x
        """
        return self.op.adjoint(self.op.forward(x)) + self.alpha*x
    
    def solve(self, y, return_series=False, return_tol=False, verbose=False):
        """ Compute solution
        :param y: input data (np.array)
        :param return_series: return the solutions for the individual iterations
        :param return_tol: return tol for the individual iterations
        :param verbose: boolean to turn on/off debug print
        :return: return specified values
        """
        x0 = self.op.adjoint(y) # a
        x = np.zeros_like(x0) # b_approx^(0)
        r = x0.copy() 
        p = r.copy()
        rr = self.normSquared(r)
        x0x0 = self.normSquared(x0) #a^Ha
        it = 0
        
        recons = []
        delta = [rr/x0x0]
        
        while rr/x0x0 > self.tol and it < self.max_iter:
            q = self.__systemMatrix__(p) # q
            tmp = rr / self.complexDot(p, q) # helper var
            x += tmp*p.copy()
            r -= tmp*q.copy()
            p = r.copy() + (self.normSquared(r)/rr)*p.copy()
            rr = self.normSquared(r)
            if verbose:
                print(it+1, rr/x0x0, self.tol)
            it += 1
            recons.append(x.copy())
            delta.append(rr/x0x0)
        
        if return_series and return_tol:
            return np.asarray(recons), np.asarray(delta)
        elif return_tol:
            return x, np.asarray(delta)
        else:
            return x
