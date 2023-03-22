"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

from ..optimization.base_optimizer import BaseReconOptimizer
from .nabla_th import Nabla, NablaT
from .prox_th import prox_p, prox_r

import tqdm
import torch

class TVReconOptimizer(BaseReconOptimizer):
    def solve(self, y, max_iter):
        # setup operators
        K = Nabla(self.mode, self.beta)
        KT = NablaT(self.mode, self.beta)

        A = self.A
        AH = self.AH

        # setup constants
        L = K.L
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

