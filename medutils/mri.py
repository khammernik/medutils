"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

DICOM_OFFSET = 0

def rss(img, coil_axis=-1):
    """ Compute root-sum-of-squares reconstruction
    :param img: input image (np.array)
    :param coil_axis: coil dimension
    :return: root-sum-of-squares reconstruction
    """
    return np.sqrt(np.sum(np.abs(img)**2, coil_axis))

def fft2c(img, axes=(-2, -1)):
    """ Compute centered and scaled fft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the fft2 is computed
    :return: centered and scaled fft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]
    
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes) / np.sqrt(img.shape[axes[0]]*img.shape[axes[1]])

def ifft2c(img, axes=(-2, -1)):
    """ Compute centered and scaled ifft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the ifft2 is computed
    :return: centered and scaled ifft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(img, axes=axes), axes=axes), axes=axes) * np.sqrt(img.shape[axes[0]]*img.shape[axes[1]])

def fft2(img, axes=(-2, -1)):
    """ Compute scaled fft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the fft2 is computed
    :return: centered and scaled fft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.fft2(img, axes=axes) / np.sqrt(img.shape[axes[0]]*img.shape[axes[1]])

def ifft2(img, axes=(-2, -1)):
    """ Compute scaled ifft2
    :param img: input image (np.array)
    :param axes: tuple of axes over which the ifft2 is computed
    :return: centered and scaled ifft2
    """
    assert len(axes) == 2
    axes = list(axes)
    full_axes = list(range(0, img.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return np.fft.ifft2(img, axes=axes) * np.sqrt(img.shape[axes[0]]*img.shape[axes[1]])

def mriAdjointOp(kspace, smaps, mask, fft_axes=(-2,-1), coil_axis=-3):
    """ Compute Cartesian MRI adjoint operation (2D)
    :param kspace: input kspace (np.array)
    :param smaps: precomputed sensitivity maps (np.array)
    :param mask: undersampling mask
    :param fft_axes: axes over which 2D fft is performed
    :param coil_axis: defines the axis of the coils (and extended coil sensitivity maps if softSENSE is used)
    :return: reconstructed image (np.array)
    """
    assert kspace.ndim >= 3
    assert kspace.ndim == smaps.ndim
    assert kspace.ndim == mask.ndim or mask.ndim == 2

    return np.sum(ifft2c(kspace * mask, axes=fft_axes)*np.conj(smaps), axis=coil_axis)

def mriForwardOp(img, smaps, mask, fft_axes=(-2,-1), soft_sense_dim=None):
    """ Compute Cartesian MRI forward operation (2D)
    :param img: input image (np.array)
    :param smaps: precomputed sensitivity maps (np.array)
    :param mask: undersampling mask
    :param fft_axes: axes over which 2D fft is performed
    :param soft_sense_dim: defines the dimension of the extended coil sensitivity maps for softSENSE
    :return: kspace (np.array)
    """
    assert img.ndim <= smaps.ndim
    kspace = fft2c(smaps * img, axes=fft_axes)*mask
    if soft_sense_dim != None:
        return np.sum(kspace, axis=soft_sense_dim)

def mriAdjointOpNoShift(kspace, smaps, mask, fft_axes=(-2,-1), coil_axis=-3):
    """ Compute Cartesian MRI adjoint operation (2D) without (i)fftshifts
    :param kspace: input kspace (pre-shifted) (np.array)
    :param smaps: precomputed sensitivity maps (np.array)
    :param mask: undersampling mask (pre-shifted)
    :param fft_axes: axes over which 2D fft is performed
    :param coil_axis: defines the axis of the coils (and extended coil sensitivity maps if softSENSE is used)
    :return: reconstructed image (np.array)
    """
    assert kspace.ndim >= 3
    assert kspace.ndim == smaps.ndim
    assert kspace.ndim == mask.ndim or mask.ndim == 2

    return np.sum(ifft2(kspace * mask, axes=fft_axes)*np.conj(smaps), axis=coil_axis)

def mriForwardOpNoShift(img, smaps, mask, fft_axes=(-2,-1), soft_sense_dim=None):
    """ Compute Cartesian MRI forward operation (2D) without (i)fftshifts
    :param img: input image (np.array)
    :param smaps: precomputed sensitivity maps (np.array)
    :param mask: undersampling mask (pre-shifted)
    :param fft_axes: axes over which 2D fft is performed
    :param soft_sense_dim: defines the dimension of the extended coil sensitivity maps for softSENSE
    :return: kspace (np.array)
    """
    assert img.ndim <= smaps.ndim
    kspace = fft2(smaps * img, axes=fft_axes)*mask
    if soft_sense_dim != None:
        return np.sum(kspace, axis=soft_sense_dim)

def estimateIntensityNormalization(img):
    """ Estimate intensity normalization based on the maximum values in the image.
    :param img: input image (np.array)
    :return: normalization value
    """
    assert img.dtype==np.float32 or img.dtype == np.float64
    # consider 10% values
    nr_values = int(np.round(img.size*0.1))

    # sort flattened image
    img_sort = np.sort(img, axis=None)

    # return median of highest 10% intensity values
    return np.median(img_sort[-nr_values:])

def removeFEOversampling(src, axes=(-2, -1), dicom_offset=DICOM_OFFSET):
    """ Remove frequency encoding (FE) oversampling.
        This is implemented such that they match with the DICOM knee images.
    :param src: input image (np.array)
    :param axes: tuple of axes containing the [nFE, nPE] dimension.
    :param dicom_offset: y-offset to match the final DICOM images.
    :return: image where the FE is removed
    """
    assert len(axes)==2
    axes = list(axes)
    full_axes = list(range(0, src.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    assert src.ndim >= 2
    nFE, nPE = src.shape[axes[0]:axes[1]+1]
    if nFE > nPE:
        return np.take(src, np.arange(int(nFE*0.25)+dicom_offset, int(nFE*0.75)+dicom_offset), axis=axes[0])
    else:
        return src

def removePEOversampling(src, axes=(-2, -1), dicom_offset=DICOM_OFFSET):
    """ Remove phase encoding (PE) oversampling.
        This is implemented such that they match with the DICOM knee images.
    :param src: input image (np.array)
    :param axes: tuple of axes containing the [nFE, nPE] dimension.
    :param dicom_offset: y-offset to match the final DICOM images.
    :return: image where the PE is removed
    """
    assert len(axes)==2
    axes = list(axes)
    full_axes = list(range(0, src.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    assert src.ndim >= 2
    nFE, nPE = src.shape[axes[0]:axes[1]+1]
    PE_OS_crop = (nPE - nFE) / 2

    if PE_OS_crop == 0:
        return src
    else:
        return np.take(src, np.arange(int(PE_OS_crop)+dicom_offset, nPE-int(PE_OS_crop)+dicom_offset), axis=axes[1])

def postprocess(src, axes=(-2, -1), dicom_offset=DICOM_OFFSET):
    """ Postprocess image.
        Remove frequency encoding (FE) oversampling and phase encoding (PE) oversampling.
        This is implemented such that they match with the DICOM knee images.
    :param src: input image (np.array)
    :param axes: tuple of axes containing the [nFE, nPE] dimension.
    :param dicom_offset: y-offset to match the final DICOM images.
    :return: image where the FE and PE is removed
    """
    assert len(axes)==2
    axes = list(axes)
    full_axes = list(range(0, src.ndim))
    axes[0] = full_axes[axes[0]]
    axes[1] = full_axes[axes[1]]

    return removePEOversampling(removeFEOversampling(src, axes, dicom_offset=dicom_offset), axes, dicom_offset=dicom_offset)

def generateRadialTrajectory(Nread, Nspokes=1, kmax=0.5):
    """ Generate a radial trajectory
    :param Nread: number of readout steps
    :param Nspokes: number of spokes
    :param kmax: maximum k-space sampling frequency
    :return: complex-valued trajectory
    """
    tmp_trajectory = np.linspace(-1, 1, num=Nread, endpoint=False) * kmax
    trajectory = np.zeros((2, Nread, Nspokes))
    
    for n in range(Nspokes):
        phi = (np.mod(Nspokes,2) + 1) * np.pi * n / Nspokes
        kx = np.cos(phi)*tmp_trajectory
        ky = np.sin(phi)*tmp_trajectory
        trajectory[0,:,n] = kx
        trajectory[1,:,n] = ky
    return trajectory[0] + 1j*trajectory[1]

def generateSpiralTrajectory(Nread, Nturns, Nshots=1, kmax=0.5):
    """ Generate a spiral trajectory
    :param Nread: number of readout steps
    :param Nturns: number of turns
    :param Nshots: number of shots
    :param kmax: maximum k-space sampling frequency
    :return: complex-valued trajectory
    """
    tmp_trajectory = np.zeros((2, Nread))
    trajectory = np.zeros((2, Nread, Nshots))
    
    # spiral in polar coordinates
    theta = np.linspace(0, Nturns * 2 * np.pi, num=Nread, endpoint=True)
    r = kmax / (Nturns * 2 * np.pi) * theta
    
    # spiral in cartesian coordinates
    tmp_trajectory[0,:] = r * np.cos(theta)
    tmp_trajectory[1,:] = r * np.sin(theta)
    
    for n in range(Nshots):
        phi = 2 * np.pi * n / Nshots
        kx = np.cos(phi)*tmp_trajectory[0,:] - np.sin(phi)*tmp_trajectory[1,:]
        ky = np.sin(phi)*tmp_trajectory[0,:] + np.cos(phi)*tmp_trajectory[1,:]
        trajectory[0,:,n] = kx
        trajectory[1,:,n] = ky
    return trajectory[0] + 1j*trajectory[1]