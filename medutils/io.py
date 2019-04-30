"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

import os
from scipy.io import loadmat as scipy_loadmat
import h5py
import numpy as np

def loadmat(filename, variable_names=None):
    """ load mat file independent of version it is saved with.
    :param filename: mat file name
    :param variable_names: list of variable names that should be loaded
    :return: dictionary of loaded data
    """

    success = False
    exc = ''
    try:
        data = __load_mat_below_7_3(filename, variable_names)
        success = True
    except Exception as e:
        exc = e

    if not success:
        try:
            data = __load_mat_7_3(filename, variable_names)
            success = True
        except Exception as e:
            exc = e

    if success:
        return data
    else:
        raise ValueError(exc)

def __load_mat_below_7_3(filename, variable_names=None):
    """ load mat file (< v7.3) with scipy 
    :param filename: mat file name
    :param variable_names: list of variable names that should be loaded
    :return: dictionary of loaded data
    """
    matfile = scipy_loadmat(filename, variable_names=variable_names)
    data = {}
    for key in matfile.keys():
        if isinstance(matfile[key], str) or  \
           isinstance(matfile[key], list) or \
           isinstance(matfile[key], dict) or \
           key == '__header__' or key == '__globals__' or key == '__version__':
            data.update({key: matfile[key]})
        elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
            data.update({key: np.asarray(matfile[key].real + 1j*matfile[key].imag, dtype='complex128')})
        else:
            data.update({key: np.asarray(matfile[key], dtype=matfile[key].dtype)})
    return data

def __load_mat_7_3(filename, variable_names=None):
    """ load mat file (v7.3) with h5py 
    :param filename: mat file name
    :param variable_names: list of variable names that should be loaded
    :return: dictionary of loaded data
    """
    matfile = h5py.File(filename, 'r')
    data = {}
    if  variable_names == None:
        for key in matfile.keys():
            if isinstance(matfile[key], str) or  \
               isinstance(matfile[key], list) or \
               isinstance(matfile[key], dict) or \
               key == '__header__' or key == '__globals__' or key == '__version__':
                data.update({key: matfile[key]})
            elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
                data.update({key: np.transpose(np.asarray(matfile[key].value.view(np.complex), dtype='complex128'))})
            else:
                data.update({key: np.transpose(np.asarray(matfile[key].value, dtype=matfile[key].dtype))})
    else:
        for key in variable_names:
            if not key in matfile.keys():
                raise RuntimeError('Variable: "' + key + '" is not in file: '+ filename)
            if isinstance(matfile[key], str) or  \
               isinstance(matfile[key], list) or \
               isinstance(matfile[key], dict) or \
               key == '__header__' or key == '__globals__' or key == '__version__':
                data.update({key: matfile[key]})
            elif  matfile[key].dtype.names != None and 'imag' in matfile[key].dtype.names:
                data.update({key: np.transpose(np.asarray(matfile[key].value.view(np.complex), dtype='complex128'))})
            else:
                data.update({key: np.transpose(np.asarray(matfile[key].value, dtype=matfile[key].dtype))})

    return data