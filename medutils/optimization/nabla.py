"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

import numpy as np
import unittest

def forward_differences_dim(x, dim):
    base_padding = [[0,1]] + [[0,0] for _ in range(x.ndim-1)]
    padding = base_padding[-dim:] + base_padding[:-dim]
    xp = np.pad(np.take(x, np.arange(1,x.shape[dim]), axis=dim), padding, mode='edge')
    return xp - x

def forward_differences(x):    
    x_fwd = np.zeros((x.ndim,) + x.shape)
    for i in range(x.ndim):
        x_fwd[i] = forward_differences_dim(x, i)

    return x_fwd

def backward_differences_dim(x, dim):
    base_padding = [[1,1]] + [[0,0] for _ in range(x.ndim-1)]
    padding = base_padding[-dim:] + base_padding[:-dim]
    xp = np.pad(np.take(x, np.arange(0,x.shape[dim]-1), axis=dim), padding, mode='constant')
    x_bwd = np.take(xp, np.arange(0,x.shape[dim]), dim) - np.take(xp, np.arange(1,x.shape[dim]+1), dim)
    return x_bwd

def backward_differences(x):
    assert x.shape[0] == len(x.shape[1:])

    x_bwd = np.zeros(x.shape)

    for i in range(x.ndim-1):
        x_bwd[i] = backward_differences_dim(x[i], i)
    return x_bwd

def nabla(x, mode='2d', beta=None):
    dx = forward_differences(x)
    if mode in ['1d', '2d', '3d', '4d']:
        return dx
    elif mode in ['1dt', '2dt', '3dt']:
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        dx[1:] *= mu1
        dx[0] *= mu2
        return dx
    else:
        raise NotImplementedError(f'Nabla for mode {mode} not defined!')

def nablaT(x, mode='2d', beta=None):
    x_bwd = backward_differences(x)
    if mode in ['1d', '2d', '3d', '4d']:
        return np.sum(x_bwd, 0)
    elif mode in ['1dt', '2dt', '3dt']:
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        return mu2 * x_bwd[0] + mu1 * np.sum(x_bwd[1:], 0)
    else:
        raise NotImplementedError(f'Nabla for mode {mode} not defined!')

def nabla_sym(x, mode='2d', beta=None):
    assert x.ndim - 1 == x.shape[0]

    if mode == '2d':
        wyy = backward_differences_dim(x[0], 0)
        wxx = backward_differences_dim(x[1], 1)
        wyx = (backward_differences_dim(x[0], 1) + backward_differences_dim(x[1], 0)) / 2.0

        result = np.concatenate([wyy[None], wxx[None], wyx[None]])

    elif mode == '2dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt = backward_differences_dim(x[0], 0)
        wyy = backward_differences_dim(x[1], 1)
        wxx = backward_differences_dim(x[2], 2)
        wty = (mu1 * backward_differences_dim(x[0], 1) + mu2 * backward_differences_dim(x[1], 0)) / 2.0 
        wtx = (mu1 * backward_differences_dim(x[0], 2) + mu2 * backward_differences_dim(x[2], 0)) / 2.0 
        wyx = (mu1 * backward_differences_dim(x[1], 2) + mu1 * backward_differences_dim(x[2], 1)) / 2.0 
        result = np.concatenate([mu2 * wtt[None], 
                                 mu1 * wyy[None],
                                 mu1 * wxx[None],
                                 wty[None],
                                 wtx[None],
                                 wyx[None]])

    elif mode == '3dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt = backward_differences_dim(x[0], 0)
        wzz = backward_differences_dim(x[1], 1)
        wyy = backward_differences_dim(x[2], 2)
        wxx = backward_differences_dim(x[3], 3)

        wtz = (mu1 * backward_differences_dim(x[0], 1) + mu2 * backward_differences_dim(x[1], 0)) / 2.0
        wty = (mu1 * backward_differences_dim(x[0], 2) + mu2 * backward_differences_dim(x[2], 0)) / 2.0
        wtx = (mu1 * backward_differences_dim(x[0], 3) + mu2 * backward_differences_dim(x[3], 0)) / 2.0

        wzy = (mu1 * backward_differences_dim(x[1], 2) + mu1 * backward_differences_dim(x[2], 1)) / 2.0
        wzx = (mu1 * backward_differences_dim(x[1], 3) + mu1 * backward_differences_dim(x[3], 1)) / 2.0
        wyx = (mu1 * backward_differences_dim(x[2], 3) + mu1 * backward_differences_dim(x[3], 2)) / 2.0

        result = np.concatenate([mu2 * wtt[None],
                                 mu1 * wzz[None], 
                                 mu1 * wyy[None],
                                 mu1 * wxx[None],
                                 wtz[None],
                                 wty[None],
                                 wtx[None],
                                 wzy[None],
                                 wzx[None],
                                 wyx[None],
                                 ])

    elif mode == '3d':
        result = nabla_sym(x, mode='2dt', beta=(1,1))

    elif mode == '4d':
        result = nabla_sym(x, mode='3dt', beta=(1,1))

    else:
        raise NotImplementedError(f'nabla_sym for mode {mode} not defined!')

    return result

def nabla_symT(w, mode='2d', beta=None):
    assert np.sum(np.arange(1,w.ndim)) == w.shape[0]
    if mode == '2d':
        wyy, wxx, wyx = w[0], w[1], w[2]
        vy = forward_differences_dim(wyy, 0) + forward_differences_dim(wyx, 1)
        vx = forward_differences_dim(wxx, 1) + forward_differences_dim(wyx, 0)
        result = np.concatenate([vy[None], vx[None]])

    elif mode == '2dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt, wyy, wxx, wty, wtx, wyx  = w[0], w[1], w[2], w[3], w[4], w[5]
        vt = mu2 * forward_differences_dim(wtt, 0) + mu1 * forward_differences_dim(wtx, 2) + mu1 * forward_differences_dim(wty, 1)
        vy = mu1 * forward_differences_dim(wyy, 1) + mu1 * forward_differences_dim(wyx, 2) + mu2 * forward_differences_dim(wty, 0)
        vx = mu1 * forward_differences_dim(wxx, 2) + mu1 * forward_differences_dim(wyx, 1) + mu2 * forward_differences_dim(wtx, 0)
        result = np.concatenate([vt[None], vy[None], vx[None]])

    elif mode == '3dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt, wzz, wyy, wxx = w[0], w[1], w[2], w[3]
        wtz, wty, wtx, wzy, wzx, wyx  = w[4], w[5], w[6], w[7], w[8], w[9]

        vz = mu1 * forward_differences_dim(wzx, 3) + \
             mu1 * forward_differences_dim(wzy, 2) + \
             mu1 * forward_differences_dim(wzz, 1) + \
             mu2 * forward_differences_dim(wtz, 0)

        vy = mu1 * forward_differences_dim(wyx, 3) + \
             mu1 * forward_differences_dim(wyy, 2) + \
             mu1 * forward_differences_dim(wzy, 1) + \
             mu2 * forward_differences_dim(wty, 0)

        vx = mu1 * forward_differences_dim(wyx, 2) + \
             mu1 * forward_differences_dim(wxx, 3) + \
             mu1 * forward_differences_dim(wzx, 1) + \
             mu2 * forward_differences_dim(wtx, 0)

        vt = mu2 * forward_differences_dim(wtt, 0) + \
             mu1 * forward_differences_dim(wtz, 1) + \
             mu1 * forward_differences_dim(wty, 2) + \
             mu1 * forward_differences_dim(wtx, 3)

        result = np.concatenate([vt[None], vz[None], vy[None], vx[None]])


    elif mode == '3d':
        result = nabla_symT(w, mode='2dt', beta=(1,1))

    elif mode == '4d':
        result = nabla_symT(w, mode='3dt', beta=(1,1))

    else:
        raise NotImplementedError(f'Nabla for mode {mode} not defined!')

    return result

class Nabla(object):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta
        if mode in ['1dt', '2dt', '3dt']:
            assert isinstance(beta, tuple) and len(beta) == 2

    def forward(self, x):
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return nabla(x_re, mode=self.mode, beta=self.beta) + 1j * nabla(x_im, mode=self.mode, beta=self.beta)
        else:
            return nabla(x, mode=self.mode, beta=self.beta)

    @property
    def L(self):
        if self.mode == '1d':
            L = 2  # np.sqrt(4)
        elif self.mode == '2d':
            L = np.sqrt(8)
        elif self.mode == '3d':
            L = np.sqrt(12)
        elif self.mode == '4d':
            L = 4  # np.sqrt(16)
        elif self.mode == '1dt':
            L = np.sqrt(4 * (1 * self.beta[0]**2 + 1 * self.beta[1]**2))
        elif self.mode == '2dt':
            L = np.sqrt(4 * (2 * self.beta[0]**2 + 1 * self.beta[1]**2))
        elif self.mode == '3dt':
            L = np.sqrt(4 * (3 * self.beta[0]**2 + 1 * self.beta[1]**2))
        else:
            raise ValueError(f'Lipschitz constant L for mode {self.mode} not defined')
        return L

class NablaT(object):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return nablaT(x_re, mode=self.mode, beta=self.beta) + 1j * nablaT(x_im, mode=self.mode, beta=self.beta)
        else:
            return nablaT(x, mode=self.mode, beta=self.beta)


class NablaSym(object):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return nabla_sym(x_re, mode=self.mode, beta=self.beta) + 1j * nabla_sym(x_im, mode=self.mode, beta=self.beta)
        else:
            return nabla_sym(x, mode=self.mode, beta=self.beta)

class NablaSymT(object):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if np.iscomplexobj(x):
            x_re = np.ascontiguousarray(np.real(x))
            x_im = np.ascontiguousarray(np.imag(x))
            return nabla_symT(x_re, mode=self.mode, beta=self.beta) + 1j * nabla_symT(x_im, mode=self.mode, beta=self.beta)
        else:
            return nabla_symT(x, mode=self.mode, beta=self.beta)


    # if mode == '1d':
    #     dx = np.pad(x[:-1], [[1,1]], mode='constant')
    #     result = dx[:-1] - dx[1:]

    # elif mode == '2d':
    #     assert x.shape[0] == 2
    #     dx = np.pad(x[0,:,:-1], [[0, 0],[1, 1]], mode='constant')
    #     dy = np.pad(x[1,:-1], [[1, 1],[0, 0]], mode='constant')
    #     result = dx[:,:-1] - dx[:,1:] + dy[:-1] - dy[1:]

    # elif mode == '2dt':
    #     assert x.shape[0] == 3
    #     assert isinstance(beta, tuple) and len(beta) == 2
    #     mu1, mu2 = beta
    #     dt = np.pad(x[2,:-1], [[1, 1], [0, 0], [0, 0]], mode='constant')
    #     dy = np.pad(x[1,:,:-1], [[0, 0], [1, 1], [0, 0]], mode='constant')
    #     dx = np.pad(x[0,...,:-1], [[0, 0], [0, 0], [1, 1]], mode='constant')


        #result = mu1 * (dx[...,:-1] - dx[...,1:]) + mu1 * (dy[:,:-1] - dy[:,1:]) + mu2 * (dt[:-1] - dt[1:])
    #return result

#%%
class NablaTest(unittest.TestCase):
    def _test(self, x, y, A, AH):
        lhs = np.sum(AH(y) * x)
        rhs = np.sum(y * A(x))

        self.assertAlmostEqual(lhs, rhs)

    def _test_sym(self, x, y, A, AH):
        Ax = A(x)
        Ax[x.shape[0]:]*=2  # we only store the mixed derivatives once, therefore we have to consider them in the adjointness check twice
        lhs = np.sum(AH(y) * x)
        rhs = np.sum(y * Ax)

        self.assertAlmostEqual(lhs, rhs)

    def test_1d(self):
        N = 4

        np.random.seed(0)

        x = np.random.randint(low=0,high=10,size=(N))
        y = np.random.randint(low=0,high=10,size=(1,N))

        A = lambda x: nabla(x, mode='1d')
        AH = lambda x: nablaT(x, mode='1d')

        self._test(x, y, A, AH)

    def test_2d(self):
        M = 4
        N = 4

        np.random.seed(42)

        x = np.random.randn(M, N)
        y = np.random.randn(2, M, N)


        A = lambda x: nabla(x, mode='2d')
        AH = lambda x: nablaT(x, mode='2d')

        self._test(x, y, A, AH)

    def test_2dt(self):
        M = 4
        N = 4
        T = 4

        beta = (1.5, 1.2)

        np.random.seed(42)

        x = np.random.randn(T, M, N)
        y = np.random.randn(3, T, M, N)

        A = lambda x: nabla(x, mode='2dt', beta=beta)
        AH = lambda x: nablaT(x, mode='2dt', beta=beta)

        self._test(x, y, A, AH)

    def test_3d(self):
        M = 4
        N = 4
        T = 4

        np.random.seed(42)

        x = np.random.randn(T, M, N)
        y = np.random.randn(3, T, M, N)

        A = lambda x: nabla(x, mode='3d')
        AH = lambda x: nablaT(x, mode='3d')

        self._test(x, y, A, AH)

    def test_3dt(self):
        M = 4
        N = 4
        T = 4
        D = 4

        beta = (1.5, 1.2)

        np.random.seed(42)

        x = np.random.randn(T, D, M, N)
        y = np.random.randn(4, D, T, M, N)

        A = lambda x: nabla(x, mode='3dt', beta=beta)
        AH = lambda x: nablaT(x, mode='3dt', beta=beta)

        self._test(x, y, A, AH)

    def test_4d(self):
        M = 4
        N = 4
        T = 4
        D = 4

        np.random.seed(42)

        x = np.random.randn(D, T, M, N)
        y = np.random.randn(4, D, T, M, N)

        A = lambda x: nabla(x, mode='4d')
        AH = lambda x: nablaT(x, mode='4d')

        self._test(x, y, A, AH)

    def test_sym_2d(self):
        M = 4
        N = 4

        np.random.seed(42)

        x = np.random.randn(2, M, N)
        y = np.random.randn(3, M, N)


        A = lambda x: nabla_sym(x, mode='2d')
        AH = lambda x: nabla_symT(x, mode='2d')

        self._test_sym(x, y, A, AH)

    def test_sym_3d(self):
        M = 4
        N = 4
        D = 4

        np.random.seed(42)

        x = np.random.randn(3, D, M, N)
        y = np.random.randn(6, D, M, N)


        A = lambda x: nabla_sym(x, mode='3d')
        AH = lambda x: nabla_symT(x, mode='3d')

        self._test_sym(x, y, A, AH)

    def test_sym_2dt(self):
        M = 4
        N = 4
        T = 4

        np.random.seed(42)

        beta = (1.2, 1.5)

        x = np.random.randn(3, T, M, N)
        y = np.random.randn(6, T, M, N)


        A = lambda x: nabla_sym(x, mode='2dt', beta=beta)
        AH = lambda x: nabla_symT(x, mode='2dt', beta=beta)

        self._test_sym(x, y, A, AH)

    def test_sym_4d(self):
        M = 4
        N = 4
        T = 4
        D = 4

        np.random.seed(42)

        x = np.random.randn(4, T, D, M, N)
        y = np.random.randn(10, T, D, M, N)


        A = lambda x: nabla_sym(x, mode='4d')
        AH = lambda x: nabla_symT(x, mode='4d')

        self._test_sym(x, y, A, AH)

    def test_sym_3dt(self):
        M = 4
        N = 4
        T = 4
        D = 4

        np.random.seed(42)

        beta = (1.2, 1.5)

        x = np.random.randn(4, T, D, M, N)
        y = np.random.randn(10, T, D, M, N)


        A = lambda x: nabla_sym(x, mode='3dt', beta=beta)
        AH = lambda x: nabla_symT(x, mode='3dt', beta=beta)

        self._test_sym(x, y, A, AH)

if __name__ == "__main__":
    unittest.run()