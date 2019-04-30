"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

import numpy as np
import skimage.measure as sm

def nrmse(img, ref, axes = (0,1)):
    """ Compute the normalized root mean squared error (nrmse)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the nrmse is computed
    :return: (mean) nrmse
    """
    nominator = np.real(np.sum( (img - ref) * np.conj(img - ref), axis = axes))
    denominator = np.real(np.sum( ref * np.conj(ref), axis = axes))
    nrmse = np.sqrt(nominator / denominator)
    return np.mean(nrmse)

def nmse(img, ref, axes = (0,1)):
    """ Compute the normalized mean squared error (nmse)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the nmse is computed
    :return: (mean) nmse
    """
    nominator = np.real(np.sum( (img - ref) * np.conj(img - ref), axis = axes))
    denominator = np.real(np.sum(ref * np.conj(ref), axis = axes))
    nmse = nominator / denominator
    return np.mean(nmse)

def nrmseAbs(img, ref, axes = (0,1)):
    """ Compute the normalized root mean squared error (nrmse) on the absolute value
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the nrmseAbs is computed
    :return: (mean) nrmseAbs
    """
    img = np.abs(img.copy())
    ref = np.abs(ref.copy())
    nominator = np.real(np.sum( (img - ref) * np.conj(img - ref), axis = axes))
    denominator = np.real(np.sum( ref * np.conj(ref), axis = axes))
    nrmse = np.sqrt(nominator / denominator)
    return np.mean(nrmse)

def mseAbs(img, ref, axes = (0,1)):
    """ Compute the mean squared error on the absolute value (mse)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the mse is computed
    :return: (mean) mse
    """
    img = np.abs(img)
    ref = np.abs(ref)
    mse = np.real(np.sum( (img - ref) * np.conj(img - ref), axis = axes))
    return np.mean(mse)

def mse(img, ref, axes=(0,1)):
    """ Compute the mean squared error (mse)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the mse is computed
    :return: (mean) mse
    """
    mse = np.real(np.sum( (img - ref) * np.conj(img - ref), axis = axes))
    return np.mean(mse)

def psnr(img, ref, axes = (0, 1), max_intensity=None):
    """ Compute the peak signal to noise ratio (psnr)
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param axes: tuple of axes over which the psnr is computed
    :param max_intensity: maximum intensity in the image. If it is None, the maximum value of :ref: is taken.
    :return: (mean) psnr
    """
    mse = np.mean(np.abs(np.abs(img) - np.abs(ref))**2, axis = axes)
    max_intensity = np.max(np.abs(ref)) if max_intensity == None else max_intensity
    mse = 10 * np.log10(max_intensity ** 2 / mse)
    return np.mean(mse)

def ssim(img, ref, dynamic_range=None, axes=(0,1)):
    """ Compute the structural similarity index.
    :param img: input image (np.array)
    :param ref: reference image (np.array)
    :param dynamic_range: If dynamic_range != None, the same given dynamic range will be used for all slices in the volume.
                          Otherwise, the dynamic_range is computed slice-per-slice.
    :param axes: tuple of axes over which the ssim is computed
    :return: (mean) ssim
    """

    assert len(axes) == 2
    assert img.shape == ref.shape
    if img.ndim == 2 and axes==(0,1):
        img = img.copy()[np.newaxis]
        ref = ref.copy()[np.newaxis]
    elif img.ndim == 2 and axes != (0,1):
        raise ValueError("axes of 2d array have to equal (0,1)")
    else:
        axes = list(axes)
        full_axes = list(range(0, img.ndim))
        transpose_axes = [item for item in full_axes if item not in axes] + axes
        unwrap_axes = [transpose_axes.index(item) for item in full_axes]
        img = np.transpose(img.copy(), transpose_axes)
        img = np.reshape(img, (np.prod(img.shape[:-2]),) + img.shape[-2:])
        ref = np.transpose(ref.copy(), transpose_axes)
        ref = np.reshape(ref, (np.prod(ref.shape[:-2]),) + ref.shape[-2:])     

    # ssim averaged over slices
    ssim_slices = []
    ref_abs = np.abs(ref)
    img_abs = np.abs(img)
    
    for i in range(ref_abs.shape[0]):
        if dynamic_range == None:
            drange = np.max(ref_abs[i]) - np.min(ref_abs[i])
        else:
            drange = dynamic_range
        _, ssim_i = sm.compare_ssim(img_abs[i], ref_abs[i],
                                 data_range=drange,
                                 gaussian_weights=True,
                                 use_sample_covariance=False,
                                 full=True)
        ssim_slices.append(np.mean(ssim_i))

    return np.mean(ssim_slices)