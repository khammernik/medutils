"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

from ..optimization.base_optimizer import BaseOptimizer
from .nabla_th import Nabla, NablaT
from .prox_th import prox_p, prox_denoise

import tqdm
import torch

class TVOptimizer(BaseOptimizer):
    def __init__(self, mode, lambd, beta=None, tau=None, prox_h=prox_denoise):
        super().__init__(mode, lambd, beta, tau)
        self.prox_h = prox_h

    def solve(self, y, max_iter):
        # setup operators
        K = Nabla(self.mode, self.beta)
        KT = NablaT(self.mode, self.beta)

        # setup constants
        L = K.L
        if self.tau != None:
            tau = self.tau
        else:
            tau = 1.0 / L
        sigma = 1.0 / (L**2 * tau)

        theta = 1.0

        # setup dual variables
        p = torch.zeros_like(K(y), device=y.device)

        # setup primal variables
        x = y.clone()
        x_bar = y.clone()

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p
            p.add_(sigma * K(x_bar))
            # prox of p
            p = prox_p(p)

            # primal update of x
            x_new = torch.sub(x, tau * KT(p))
            # prox of x
            x_new = self.prox_h(x_new, y, tau * self.lambd)

            # over-relaxation
            x_bar = x_new + theta * (x_new - x)
            x.copy_(x_new)

        return x
