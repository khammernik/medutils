"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

from ..optimization.base_optimizer import BaseReconOptimizer
from .nabla_th import Nabla, NablaT, NablaSym, NablaSymT
from .prox_th import prox_p, prox_q, prox_r
from ..optimization.stepsize import adapt_stepsize

import tqdm
import torch
import numpy as np
from functools import partial

class TVReconOptimizer(BaseReconOptimizer):
    """ Total Variation

    Chambolle & Pock. A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging.
    Journal of Mathematical Imaging and Vision 40, 120-145, 2011.
    https://doi.org/10.1007/s10851-010-0251-1
    """
    def solve(self, y, max_iter):
        # setup operators
        K = Nabla(self.mode, self.beta) if self.K is None else partial(self.K, mode=self.mode, beta=self.beta)
        KT = NablaT(self.mode, self.beta) if self.KT is None else partial(self.KT, mode=self.mode, beta=self.beta)

        A = self.A
        AH = self.AH

        # setup constants
        L = Nabla(self.mode, self.beta).L         # ToDo: Adjust to input K
        if self.tau != None:
            tau = self.tau
        else:
            tau = 1.0 / L
        sigma = 1.0 / (L**2 * tau)

        theta = 1.0

        # setup primal variables
        x = AH(y).clone()
        x_bar = x.clone()
        
        # setup dual variables
        p = torch.zeros_like(K(x), device=y.device)
        r = torch.zeros_like(y, device=y.device)

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p
            p.add_(sigma * K(x_bar))
            # prox of p
            p = prox_p(p)

            # dual update of r
            r.add_(sigma * (A(x_bar) - y))
            r = prox_r(r, sigma/self.lambd)

            # primal update of x
            x_new = x - tau * (KT(p) + AH(r))

            # over-relaxation
            x_bar = x_new + theta * (x_new - x)
            x.copy_(x_new)

        return x


class TGVReconOptimizer(BaseReconOptimizer):
    """ Total Generalized Variation (TGV)

    Bredies et al. Total Generalized Variation.
    SIAM Journal on Imaging Sciences 3(3):492-526, 2010.
    https://doi.org/10.1137/090769521


    Knoll et al. Second order total generalized variation (TGV) for MRI.
    Magnetic Resonance in Medicine 65(2):480-4941, 2011.
    https://doi.org/10.1002/mrm.22595
    """
    def __init__(self, A, AH, mode, lambd, alpha0, alpha1, beta=None):
        super().__init__(A, AH, mode, lambd, beta)
        self.alpha0 = alpha0
        self.alpha1 = alpha1

    def solve(self, y, max_iter):
        # setup operator # ToDo: Adapt to custom operators
        K = Nabla(self.mode, self.beta)
        KT = NablaT(self.mode, self.beta)
        E = NablaSym(self.mode, self.beta)
        ET = NablaSymT(self.mode, self.beta)

        A = self.A
        AH = self.AH

        # setup constants
        # TODO adapt to TGV operator
        L = Nabla(self.mode, self.beta).L
        if self.tau != None:
            tau = self.tau
        else:
            tau = 1.0 / L
        sigma = 1.0 / (L**2 * tau)

        theta = 1.0

        # setup primal variables
        x = AH(y).clone()
        x_bar = x.clone()
        v = torch.zeros_like(K(x), device=y.device)
        v_bar = v.clone()

        # setup dual variables
        p = torch.zeros_like(v, device=y.device)
        q = torch.zeros_like(E(p), device=y.device)
        r = torch.zeros_like(y, device=y.device)

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p
            p.add_(sigma * (K(x_bar) - v_bar))
            # prox of p
            p = prox_p(p, self.alpha1)
            
            # dual update of r
            r.add_(sigma * (A(x_bar) - y))
            r = prox_r(r, sigma / self.lambd)

            # dual update of q
            q.add_(sigma * E(v_bar))
            # prox of q
            q = prox_q(q, self.ndim, self.alpha0)

            # primal update of x
            x_new = x - tau * (KT(p) + AH(r))

            # primal update of v
            v_new = v - tau*(ET(q) - p)

            # over-relaxation of x
            x_bar = x_new + theta * (x_new - x)
            x = x_new.clone()

            # over-relaxation of v
            v_bar = v_new + theta * (v_new - v)
            v = v_new.clone()

        return x

class ICTVReconOptimizer(BaseReconOptimizer):
    """ Infimal Convolution of Total Variation (ICTV)

    Holler and Kunisch. On Infimal Convolution of TV-Type Functionals and Applications to Video and Image Reconstruction
    SIAM Journal on Imaging Sciences 7(4):2258-2300, 2014
    https://doi.org/10.1137/130948793

    Schloegl et al. Infimal convolution of total generalized variation functionals for dynamic MRI.
    Magnetic Resonance in Medicine 78(1):142-155, 2017.
    https://doi.org/10.1002/mrm.26352
    """
    def __init__(self, A, AH, mode, lambd, alpha1, s, beta1=None, beta2=None):
        super().__init__(A, AH, mode, lambd)
        self.alpha1 = alpha1
        self.s = s
        assert 0 <= s <= 1
        self.beta1 = beta1
        self.beta2 = beta2
        
    def solve(self, y, max_iter):
        # setup operators # ToDo: Adapt to custom operators
        K_beta1 = Nabla(self.mode, self.beta1) if self.K is None else partial(self.K, mode=self.mode, beta=self.beta1)
        KT_beta1 = NablaT(self.mode, self.beta1) if self.KT is None else partial(self.KT, mode=self.mode, beta=self.beta1)
        K_beta2 = Nabla(self.mode, self.beta2) if self.K is None else partial(self.KT, mode=self.mode, beta=self.beta2)
        KT_beta2 = NablaT(self.mode, self.beta2) if self.KT is None else partial(self.KT, mode=self.mode, beta=self.beta2)

        A = self.A
        AH = self.AH

        # setup constants
        # adaptive stepsize scheme according to Schloegl et al.
        tau = 10.0
        sigma = 10.0

        theta = 1.0

        gamma1 = self.s / np.minimum(self.s, 1 - self.s)
        gamma2 = (1 - self.s) / np.minimum(self.s, 1 - self.s)

        # setup primal variables
        x = AH(y).clone()
        x_bar = x.clone()
        v = torch.zeros_like(x, device=x.device)
        v_bar = v.clone()

        # setup dual variables
        p1 = torch.zeros_like(K_beta1(x), device=x.device)
        p2 = torch.zeros_like(K_beta1(x), device=x.device)
        r = torch.zeros_like(y, device=y.device)

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p1
            p1.add_(sigma * K_beta1(x_bar - v_bar))
            # prox of p1
            p1 = prox_p(p1, gamma1 * self.alpha1)

            # dual update of p2
            p2.add_(sigma * K_beta2(v_bar))
            # prox of p1
            p2 = prox_p(p2, gamma2 * self.alpha1)

            # dual update of r
            r.add_(sigma * (A(x_bar) - y))
            r = prox_r(r, sigma / self.lambd)

            # primal update x
            x_new = x - tau * (KT_beta1(p1) + AH(r))

            # primal update v
            v_new = v - tau * (KT_beta2(p2) - KT_beta1(p1))

            # adapt step-size
            x1 = x_new - x
            x2 = v_new - v
            x_all = torch.cat([x1.flatten(), x2.flatten()])

            # step size update
            Kx1 = K_beta1(x1) - K_beta1(x2)
            Kx2 = K_beta2(x2)
            Kx3 = A(x1)
            Kx_all = torch.cat([Kx1.flatten(), Kx2.flatten(), Kx3.flatten()])

            nominator = x_all.abs().pow_(2).sum().sqrt_().detach().cpu().numpy()
            denominator = Kx_all.abs().pow_(2).sum().sqrt_().detach().cpu().numpy()

            sigma = adapt_stepsize(sigma, tau, theta, nominator/denominator)
            tau = sigma

            # over-relaxation
            x_bar = x_new + theta * (x_new - x)
            x = x_new.clone()

            v_bar = v_new + theta * (v_new - v)
            v = v_new.clone()

        return x, v

class ICTGVReconOptimizer(BaseReconOptimizer):
    """ Infimal Convolution of Total Generalized Variation (ICTGV)

    Holler and Kunisch. On Infimal Convolution of TV-Type Functionals and Applications to Video and Image Reconstruction
    SIAM Journal on Imaging Sciences 7(4):2258-2300, 2014
    https://doi.org/10.1137/130948793

    Schloegl et al. Infimal convolution of total generalized variation functionals for dynamic MRI.
    Magnetic Resonance in Medicine 78(1):142-155, 2017.
    https://doi.org/10.1002/mrm.26352
    """
    def __init__(self, A, AH, mode, lambd, alpha0, alpha1, s, beta1=None, beta2=None):
        super().__init__(A, AH, mode, lambd)
        self.alpha0 = alpha0
        self.alpha1 = alpha1
        self.s = s
        assert 0 <= s <= 1
        self.beta1 = beta1
        self.beta2 = beta2

    def solve(self, y, max_iter):
        # setup operators # ToDo: Adapt to custom operators
        K_beta1 = Nabla(self.mode, self.beta1)
        KT_beta1 = NablaT(self.mode, self.beta1)
        E_beta1 = NablaSym(self.mode, self.beta1)
        ET_beta1 = NablaSymT(self.mode, self.beta1)
        K_beta2 = Nabla(self.mode, self.beta2)
        KT_beta2 = NablaT(self.mode, self.beta2)
        E_beta2 = NablaSym(self.mode, self.beta2)
        ET_beta2 = NablaSymT(self.mode, self.beta2)

        A = self.A
        AH = self.AH

        # setup constants
        # adaptive stepsize scheme according to Schloegl et al.
        tau = 10.0
        sigma = 10.0

        theta = 1.0

        gamma1 = self.s / np.minimum(self.s, 1 - self.s)
        gamma2 = (1 - self.s) / np.minimum(self.s, 1 - self.s)

        # setup primal variables
        x = AH(y).clone()
        x_bar = x.clone()

        v = torch.zeros_like(x, device=x.device)
        v_bar = v.clone()

        w1 = torch.zeros_like(K_beta1(x), device=x.device)
        w1_bar = w1.clone()
        
        w2 = w1.clone()
        w2_bar = w2.clone()

        # setup dual variables
        p1 = w2.clone()
        p2 = w2.clone()
        q1 = torch.zeros_like(E_beta1(p1), device=p1.device)
        q2 = torch.zeros_like(E_beta1(p1), device=p1.device)
        r = torch.zeros_like(y, device=y.device)

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p1
            p1.add_(sigma * (K_beta1(x_bar - v_bar) - w1_bar))
            # prox of p1
            p1 = prox_p(p1, gamma1 * self.alpha1)
            
            # dual update of q1
            q1.add_(sigma * E_beta1(w1_bar))
            # prox of q1
            q1 = prox_q(q1, self.ndim, gamma1 * self.alpha0)

            # dual update of p2
            p2.add_(sigma * (K_beta2(v_bar) - w2_bar))
            # prox of p2
            p2 = prox_p(p2, gamma2 * self.alpha1)

            # dual update of q2
            q2.add_(sigma * E_beta2(w2_bar))
            # prox of q2
            q2 = prox_q(q2, self.ndim, gamma2 * self.alpha0)

            # dual update of r
            r.add_(sigma * (A(x_bar) - y))
            r = prox_r(r, sigma / self.lambd)

            # primal update of u
            x_new = x - tau * (KT_beta1(p1) + AH(r))

            # primal update of w1
            w1_new = w1 - tau * (ET_beta1(q1) - p1)

            # primal update of v
            v_new = v - tau * (KT_beta2(p2) - KT_beta1(p1))

            # primal update of w2
            w2_new = w2 - tau * (ET_beta2(q2) - p2)

            # step size update
            x1 = x_new - x
            x2 = w1_new - w1
            x3 = v_new - v
            x4 = w2_new - w2
            x_all = torch.cat([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()])

            Kx1 = K_beta1(x1) - x2 - K_beta1(x3)
            Kx2 = E_beta1(x2)
            Kx3 = K_beta2(x3) - x4
            Kx4 = E_beta2(x4)
            Kx5 = A(x1)
            Kx_all = torch.cat([Kx1.flatten(), Kx2.flatten(), Kx3.flatten(), Kx4.flatten(), Kx5.flatten()])

            nominator = x_all.abs().pow_(2).sum().sqrt_().detach().cpu().numpy()
            denominator = Kx_all.abs().pow_(2).sum().sqrt_().detach().cpu().numpy()

            sigma = adapt_stepsize(sigma, tau, theta, nominator/denominator)
            tau = sigma

            # over-relaxation
            x_bar = x_new + theta * (x_new - x)
            x = x_new.clone()

            v_bar = v_new + theta * (v_new - v)
            v = v_new.clone()

            w1_bar = w1_new + theta * (w1_new - w1)
            w1 = w1_new.clone()

            w2_bar = w2_new + theta * (w2_new - w2)
            w2 = w2_new.clone()

        return x, v
