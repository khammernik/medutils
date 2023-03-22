"""
This file is part of medutils.

Copyright (C) 2023 Kerstin Hammernik <k dot hammernik at tum dot de>
I31 - Technical University of Munich
https://www.kiinformatik.mri.tum.de/de/hammernik-kerstin
"""

import torch

def prox_p(p, alpha=1):
    p_norm = (torch.abs(p) ** 2).sum(dim=0, keepdims=True).sqrt_().div_(alpha).clamp_(min=1)
    return p / p_norm

def prox_q(q, ndim, alpha=1):
    q_square = torch.abs(q)**2
    q_norm = (q_square[:ndim].sum(0,keepdims=True) + 2*q_square[ndim:].sum(0, keepdims=True)).sqrt_().div_(alpha).clamp_(min=1)
    return q / q_norm

def prox_denoise(x, y, alpha):
    return (x + alpha * y).div_(1 + alpha)

def prox_r(r, alpha):
    return r / (1 + alpha)