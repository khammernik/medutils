"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

import numpy as np

class BaseOptimizer(object):
    def __init__(self, mode, lambd, beta=None, tau=None):
        self.mode = mode
        self.lambd = lambd
        self.beta = beta
        self.tau = tau
        if self.mode == '1d':
            self.ndim = 1
        elif self.mode == '2d':
            self.ndim = 2
        elif self.mode in ['2dt', '3d']:
            self.ndim = 3
        elif self.mode in ['3dt', '4d']:
            self.ndim = 4
        else:
            raise ValueError(f'ndim for mode {mode} not defined')

    def solve(self, f, max_iter):
        raise NotImplementedError

class BaseReconOptimizer(BaseOptimizer):
    def __init__(self, A, AH, mode, lambd, beta=None, tau=None):
        self.A = A
        self.AH = AH
        super().__init__(mode, lambd, beta, tau)