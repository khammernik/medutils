"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

import numpy as np

def prox_p(p, alpha=1):
    p_norm = np.sqrt(np.sum(np.abs(p) ** 2, 0, keepdims=True))
    return p / np.maximum(1, p_norm/alpha)

def prox_q(q, ndim, alpha=1):
    q_square = np.abs(q)**2
    q_norm = np.sqrt(np.sum(q_square[:ndim],0,keepdims=True) + 2*np.sum(q_square[ndim:], 0, keepdims=True))
    return q / np.maximum(1, q_norm/alpha)

def prox_denoise(x, y, alpha):
    return (x + alpha * y) / (1 + alpha)

def prox_r(r, alpha):
    return r / (1 + alpha)