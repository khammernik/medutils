"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

from .base_optimizer import BaseOptimizer
from .nabla import Nabla, NablaT
from .prox import prox_p, prox_denoise

import numpy as np
import tqdm

class TVOptimizer(BaseOptimizer):
    def __init__(self, mode, lambd, beta=None, tau=None, prox_h=prox_denoise):
        super().__init__(mode, lambd, beta, tau)
        self.prox_h = prox_h

    def solve(self, y, max_iter):
        # setup operators
        op = Nabla(self.mode, self.beta)
        K = lambda x: op.forward(x)
        KT = lambda x: NablaT(self.mode, self.beta).forward(x)

        # setup constants
        L = op.L
        if self.tau != None:
            tau = self.tau
        else:
            tau = 1.0 / L
        sigma = 1.0 / (L**2 * tau)

        theta = 1.0

        # setup dual variables
        p = np.zeros_like(K(y))

        # setup primal variables
        x = y.copy()
        x_bar = y.copy()

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p
            p = p.copy() + sigma * K(x_bar)
            # prox of p
            p = prox_p(p)

            # primal update of x
            x_new = x.copy() - tau * KT(p)
            # prox of x
            x_new = self.prox_h(x_new, y, tau * self.lambd)

            # over-relaxation
            x_bar = x_new.copy() + theta * (x_new.copy() - x.copy())
            x = x_new.copy()

        return x
