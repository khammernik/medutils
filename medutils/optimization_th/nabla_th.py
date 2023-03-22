"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

import torch
import numpy as np
import unittest

def forward_differences_dim(x, dim):
    base_padding = [[1,0]] + [[0,0] for _ in range(x.ndim-1)]
    padding = base_padding[-dim:] + base_padding[:-dim]
    padding = [p for ppair in padding for p in ppair]
    xc = torch.index_select(x, dim, torch.arange(1,x.shape[dim], device=x.device))

    if xc.ndim == 1:
        padding = [0,0] + padding
        xp = torch.nn.functional.pad(xc.view(1,1,1,-1), padding[::-1], mode='replicate')[0,0,0]
    elif xc.ndim in [2,3]:
        xp = torch.nn.functional.pad(xc.view(1,1,*xc.shape), padding[::-1], mode='replicate')[0,0]
    else:
        raise NotImplementedError("Pytorch doesn't support 4D padding for now.")
    return xp - x

def forward_differences(x):    
    x_fwd = torch.zeros((x.ndim,) + x.shape, device=x.device, dtype=x.dtype)
    for i in range(x.ndim):
        x_fwd[i] = forward_differences_dim(x, i)

    return x_fwd

def backward_differences_dim(x, dim):
    base_padding = [[1,1]] + [[0,0] for _ in range(x.ndim-1)]
    padding = base_padding[-dim:] + base_padding[:-dim]
    padding = [p for ppair in padding for p in ppair]
    xc = torch.index_select(x, dim, torch.arange(0,x.shape[dim]-1, device=x.device))
    if xc.ndim == 1:
        padding = [0,0] + padding
        xp = torch.nn.functional.pad(xc.view(1,1,1,-1), padding[::-1], mode='constant')[0,0,0]
    elif xc.ndim in [2,3]:
        xp = torch.nn.functional.pad(xc.view(1,1,*xc.shape), padding[::-1], mode='constant')[0,0]
    else:
        raise NotImplementedError("Pytorch doesn't support >= 4D padding for now.")

    x_bwd = torch.index_select(xp, dim, torch.arange(0,x.shape[dim], device=x.device)) - torch.index_select(xp, dim, torch.arange(1,x.shape[dim]+1, device=x.device))
    return x_bwd

def backward_differences(x):
    assert x.shape[0] == len(x.shape[1:])

    x_bwd = torch.zeros(x.shape, device=x.device, dtype=x.dtype)

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
        return torch.sum(x_bwd, 0)
    elif mode in ['1dt', '2dt', '3dt']:
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        return mu2 * x_bwd[0] + mu1 * torch.sum(x_bwd[1:], 0)
    else:
        raise NotImplementedError(f'Nabla for mode {mode} not defined!')

def nabla_sym(x, mode='2d', beta=None):
    assert x.ndim - 1 == x.shape[0]

    if mode == '2d':
        wyy = backward_differences_dim(x[0], 0)
        wxx = backward_differences_dim(x[1], 1)
        wyx = (backward_differences_dim(x[0], 1) + backward_differences_dim(x[1], 0)) / 2.0

        result = torch.cat([wyy[None], wxx[None], wyx[None]])

    elif mode == '2dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt = backward_differences_dim(x[0], 0)
        wyy = backward_differences_dim(x[1], 1)
        wxx = backward_differences_dim(x[2], 2)
        wty = (mu1 * backward_differences_dim(x[0], 1) + mu2 * backward_differences_dim(x[1], 0)) / 2.0 
        wtx = (mu1 * backward_differences_dim(x[0], 2) + mu2 * backward_differences_dim(x[2], 0)) / 2.0 
        wyx = (mu1 * backward_differences_dim(x[1], 2) + mu1 * backward_differences_dim(x[2], 1)) / 2.0 
        result = torch.cat([mu2 * wtt[None], 
                                 mu1 * wyy[None],
                                 mu1 * wxx[None],
                                 wty[None],
                                 wtx[None],
                                 wyx[None]])
    elif mode == '3d':
        result = nabla_sym(x, mode='2dt', beta=(1,1))

    else:
        raise NotImplementedError(f'nabla_sym for mode {mode} not defined!')

    return result


def nabla_symT(w, mode='2d', beta=None):
    assert torch.sum(torch.arange(1,w.ndim)) == w.shape[0]
    if mode == '2d':
        wyy, wxx, wyx = w[0], w[1], w[2]
        vy = forward_differences_dim(wyy, 0) + forward_differences_dim(wyx, 1)
        vx = forward_differences_dim(wxx, 1) + forward_differences_dim(wyx, 0)
        result = torch.cat([vy[None], vx[None]])

    elif mode == '2dt':
        assert isinstance(beta, tuple) and len(beta) == 2
        mu1, mu2 = beta
        wtt, wyy, wxx, wty, wtx, wyx  = w[0], w[1], w[2], w[3], w[4], w[5]
        vt = mu2 * forward_differences_dim(wtt, 0) + mu1 * forward_differences_dim(wtx, 2) + mu1 * forward_differences_dim(wty, 1)
        vy = mu1 * forward_differences_dim(wyy, 1) + mu1 * forward_differences_dim(wyx, 2) + mu2 * forward_differences_dim(wty, 0)
        vx = mu1 * forward_differences_dim(wxx, 2) + mu1 * forward_differences_dim(wyx, 1) + mu2 * forward_differences_dim(wtx, 0)
        result = torch.cat([vt[None], vy[None], vx[None]])

    elif mode == '3d':
        result = nabla_symT(w, mode='2dt', beta=(1,1))

    else:
        raise NotImplementedError(f'Nabla for mode {mode} not defined!')

    return result

class Nabla(torch.nn.Module):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if x.is_complex():
            out_re = nabla(torch.real(x).contiguous(), mode=self.mode, beta=self.beta)
            out_im = nabla(torch.imag(x).contiguous(), mode=self.mode, beta=self.beta)
            out = torch.complex(out_re, out_im)
        else:
            out = nabla(x, mode=self.mode, beta=self.beta)
        return out

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

class NablaT(torch.nn.Module):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if x.is_complex():
            out_re = nablaT(torch.real(x).contiguous(), mode=self.mode, beta=self.beta)
            out_im = nablaT(torch.imag(x).contiguous(), mode=self.mode, beta=self.beta)
            out = torch.complex(out_re, out_im)
        else:
            out = nablaT(x, mode=self.mode, beta=self.beta)
        return out

class NablaSym(torch.nn.Module):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if x.is_complex():
            out_re = nabla_sym(torch.real(x).contiguous(), mode=self.mode, beta=self.beta)
            out_im = nabla_sym(torch.imag(x).contiguous(), mode=self.mode, beta=self.beta)
            out = torch.complex(out_re, out_im)
        else:
            out = nabla_sym(x, mode=self.mode, beta=self.beta)
        return out

class NablaSymT(torch.nn.Module):
    def __init__(self, mode='2d', beta=None) -> None:
        super().__init__()
        self.mode = mode
        self.beta = beta

    def forward(self, x):
        if x.is_complex():
            out_re = nabla_symT(torch.real(x).contiguous(), mode=self.mode, beta=self.beta)
            out_im = nabla_symT(torch.imag(x).contiguous(), mode=self.mode, beta=self.beta)
            out = torch.complex(out_re, out_im)
        else:
            out = nabla_symT(x, mode=self.mode, beta=self.beta)
        return out

#%%
class NablaTest(unittest.TestCase):
    def _test(self, x, y, A, AH):
        x = x.float().cuda()
        y = y.float().cuda()

        lhs = torch.sum(AH(y) * x).detach().cpu().numpy()
        rhs = torch.sum(y * A(x)).detach().cpu().numpy()

        self.assertAlmostEqual(lhs, rhs, places=5)

    def _test_sym(self, x, y, A, AH):
        x = x.float().cuda()
        y = y.float().cuda()

        Ax = A(x)
        Ax[x.shape[0]:]*=2  # we only store the mixed derivatives once, therefore we have to consider them in the adjointness check twice
        lhs = torch.sum(AH(y) * x).detach().cpu().numpy()
        rhs = torch.sum(y * Ax).detach().cpu().numpy()

        self.assertAlmostEqual(lhs, rhs, places=5)

    def test_1d(self):
        N = 4

        torch.manual_seed(0)

        x = torch.randint(low=0,high=10,size=(N,))
        y = torch.randint(low=0,high=10,size=(1,N))

        A = Nabla(mode='1d')
        AH = NablaT(mode='1d')

        self._test(x, y, A, AH)

    def test_2d(self):
        M = 4
        N = 4

        torch.manual_seed(42)

        x = torch.randn(M, N)
        y = torch.randn(2, M, N)


        A = Nabla(mode='2d')
        AH = NablaT(mode='2d')

        self._test(x, y, A, AH)

    def test_2dt(self):
        M = 4
        N = 4
        T = 4

        beta = (1.5, 1.2)

        torch.manual_seed(42)

        x = torch.randn(T, M, N)
        y = torch.randn(3, T, M, N)

        A = Nabla(mode='2dt', beta=beta)
        AH = NablaT(mode='2dt', beta=beta)
        self._test(x, y, A, AH)

    def test_3d(self):
        M = 4
        N = 4
        T = 4

        torch.manual_seed(42)

        x = torch.randn(T, M, N)
        y = torch.randn(3, T, M, N)

        A = Nabla(mode='3d')
        AH = NablaT(mode='3d')

        self._test(x, y, A, AH)

    # def test_3dt(self):
    #     M = 4
    #     N = 4
    #     T = 4
    #     D = 4

    #     beta = (1.5, 1.2)

    #     torch.manual_seed(42)

    #     x = torch.randn(T, D, M, N)
    #     y = torch.randn(4, D, T, M, N)

    #     A = Nabla(mode='3dt', beta=beta)
    #     AH = NablaT(mode='3dt', beta=beta)

    #     self._test(x, y, A, AH)

    # def test_4d(self):
    #     M = 4
    #     N = 4
    #     T = 4
    #     D = 4

    #     torch.manual_seed(42)

    #     x = torch.randn(D, T, M, N)
    #     y = torch.randn(4, D, T, M, N)

    #     A = Nabla(mode='4d')
    #     AH = NablaT(mode='4d')

    #     self._test(x, y, A, AH)

    def test_sym_2d(self):
        M = 4
        N = 4

        torch.manual_seed(42)

        x = torch.randn(2, M, N)
        y = torch.randn(3, M, N)


        A = NablaSym(mode='2d')
        AH = NablaSymT(mode='2d')

        self._test_sym(x, y, A, AH)

    def test_sym_3d(self):
        M = 4
        N = 4
        D = 4

        torch.manual_seed(42)

        x = torch.randn(3, D, M, N)
        y = torch.randn(6, D, M, N)


        A = NablaSym(mode='3d')
        AH = NablaSymT(mode='3d')

        self._test_sym(x, y, A, AH)

    def test_sym_2dt(self):
        M = 4
        N = 4
        T = 4

        torch.manual_seed(42)

        beta = (1.2, 1.1)

        x = torch.randn(3, T, M, N)
        y = torch.randn(6, T, M, N)


        A = NablaSym(mode='2dt', beta=beta)
        AH = NablaSymT(mode='2dt', beta=beta)

        self._test_sym(x, y, A, AH)

    # def test_sym_4d(self):
    #     M = 4
    #     N = 4
    #     T = 4
    #     D = 4

    #     torch.manual_seed(42)

    #     x = torch.randn(4, T, D, M, N)
    #     y = torch.randn(10, T, D, M, N)


        # A = NablaSym(mode='4d')
        # AH = NablaSymT(mode='4d')

    #     self._test_sym(x, y, A, AH)

    # def test_sym_3dt(self):
    #     M = 4
    #     N = 4
    #     T = 4
    #     D = 4

    #     torch.manual_seed(42)

    #     beta = (1.2, 1.5)

    #     x = torch.randn(4, T, D, M, N)
    #     y = torch.randn(10, T, D, M, N)


        # A = NablaSym(mode='3dt', beta=beta)
        # AH = NablaSymT(mode='3dt', beta=beta)

    #     self._test_sym(x, y, A, AH)

if __name__ == "__main__":
    unittest.run()
