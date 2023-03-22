"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

from .base_optimizer import BaseReconOptimizer
from .nabla import Nabla, NablaT
from .prox import prox_p, prox_r

import numpy as np
import tqdm

class TVReconOptimizer(BaseReconOptimizer):
    def solve(self, y, max_iter):
        # setup operators
        op = Nabla(self.mode, self.beta)
        K = lambda x: op.forward(x)
        KT = lambda x: NablaT(self.mode, self.beta).forward(x)

        A = self.A
        AH = self.AH

        # setup constants
        L = op.L
        if self.tau != None:
            tau = self.tau
        else:
            tau = 1.0 / L
        sigma = 1.0 / (L**2 * tau)

        theta = 1.0

        # setup primal variables
        x = AH(y).copy()
        x_bar = x.copy()

        # setup dual variables
        p = np.zeros_like(K(x))
        r = np.zeros_like(y)

        for _ in tqdm.tqdm(range(max_iter)):
            # dual update of p
            p = p.copy() + sigma * K(x_bar)
            # prox of p
            p = prox_p(p)

            # dual update of r
            r = r.copy() + sigma * (A(x_bar) - y)
            r = prox_r(r, sigma/self.lambd)

            # primal update of x
            x_new = x.copy() - tau * (KT(p) + AH(r))

            # over-relaxation
            x_bar = x_new.copy() + theta * (x_new.copy() - x.copy())
            x = x_new.copy()

        return x

