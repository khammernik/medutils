"""
This file is part of medutils.

Copyright (C) 2019 Kerstin Hammernik <hammernik at icg dot tugraz dot at>
Institute of Computer Graphics and Vision, Graz University of Technology
https://www.tugraz.at/institute/icg/research/team-pock/
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import scipy.misc
import os

def kshow(kspace, title="", offset=1e-4):
    """ Show k-space
    :param kspace: input k-space (np.array)
    :param title: plot title
    :param offset: offset for log scale
    """
    img = np.abs(kspace)
    img /= np.max(img)
    img = np.log(img + offset)
    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)
    plt.show()

def ksave(kspace, filepath, offset=1e-4):
    """ Save k-space
    :param kspace: input k-space (np.array)
    :param filepath: path to file where k-space should be save
    :param offset: offset for log scale
    """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    img = np.abs(kspace)
    img /= np.max(img)
    img = np.log(img + offset)
    scipy.misc.imsave(filepath, (normalize(img)).astype(np.uint8))

def imshow(img, title=""):
    """ Show (magnitude) image in grayscale
    :param img: input image (np.array)
    :param title: plot title
    """
    if np.iscomplexobj(img):
        #print('img is complex! Take absolute value.')
        img = np.abs(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(title)

def imsave(img, filepath, normalize_img=True):
    """ Save (magnitude) image in grayscale
    :param img: input image (np.array)
    :param filepath: path to file where k-space should be save
    :normalize_img: boolean if image should be normalized between [0, 255] before saving
    """
    path = os.path.dirname(filepath) or '.'
    if not os.path.exists(path):
        os.makedirs(path)

    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img)

    if normalize_img:
        img  = normalize(img)
    scipy.misc.imsave(filepath, img.astype(np.uint8))

def show(volume, vmin=None, vmax=None, title="slice", logscale=False, logoffset=1e-4, transpose=None):
    """ Interactive volume displayer.
        Press 'v' to switch to viridis colormap.
        Press 'g' to switch to grayscale.
        Scroll through slices.
        Press and move left mouse button to change [vmin, vmax].
        Press 'c' to reset [vmin, vmax].
    :param vmin: minimum value to display
    :param vmax: maximum value to display
    :param title: plot title
    :param logscale: boolean if image should be displayed in log-scale
    :param logoffset: offset for log scale
    :param transpose: tuple of transpose axis such that slice-dimension is at shape position 0.
    """
    volume = volume.copy()
    assert volume.ndim in [2, 3]
    if volume.ndim == 2:
        volume = volume[None, ...]
    if transpose != None:
        volume = volume.transpose(transpose)

    if np.iscomplexobj(volume):
        # print('img is complex! Take absolute value.')
        volume = np.abs(volume)

    if logscale:
        volume = np.log(volume + logoffset)

    cmap = 'gray'
    fig, ax = plt.subplots()
    if vmin==None:
        vmin = np.min(volume)
    if vmax==None:
        vmax = np.max(volume)

    ax.idx = 0
    ax.imshow(volume[ax.idx], vmin=vmin, vmax=vmax, cmap=cmap, interpolation='nearest')
    ax.set_title(title + f" {ax.idx}")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    max_slices = volume.shape[0] - 1
    ax.line = []

    def onScroll(event):
        event_fig = event.canvas.figure
        ax.idx = np.minimum(max_slices, np.maximum(0, ax.idx-int(event.step)))
        ax.images[0].set_array(volume[ax.idx])
        ax.set_title(title + f" {ax.idx}")
        event_fig.canvas.draw()

    def onMouseMotion(event):
        if event.button == 1:
            ax.line.append([event.x, event.y])
            if len(ax.line) > 1:
                delta_x = ax.line[-1][0] - ax.line[-2][0]
                delta_y = ax.line[-1][1] - ax.line[-2][1]
                current_vmin, current_vmax = ax.images[0].get_clim()
                current_vmin = np.maximum(vmin, np.minimum(current_vmax, current_vmin+delta_x*0.01*(vmax-vmin)))
                current_vmax = np.minimum(vmax, np.maximum(current_vmin, current_vmax+delta_y*0.01*(vmax-vmin)))
                ax.images[0].set_clim(current_vmin, current_vmax)
        elif event.button != 1:
            ax.line = []

    def onKeyPress(event):
        if event.key == 'r':
            ax.images[0].set_clim(vmin, vmax)
        elif event.key == 'g':
            ax.images[0].set_cmap('gray')
        elif event.key == 'v':
            ax.images[0].set_cmap('viridis')
        elif event.key == 'up' or event.key == 'down':
            event_step = 1 if event.key == 'up' else -1
            event_fig = event.canvas.figure
            ax.idx = np.minimum(max_slices, np.maximum(0, ax.idx-int(event_step)))
            ax.images[0].set_array(volume[ax.idx])
            ax.set_title(title + f" {ax.idx}")
            event_fig.canvas.draw()
                        
    fig.canvas.mpl_connect('scroll_event', onScroll)
    fig.canvas.mpl_connect('motion_notify_event', onMouseMotion)
    fig.canvas.mpl_connect('key_press_event', onKeyPress)

def save(data, fname, logoffset=1e-4, logscale=False, cmap='gray'):
    """ Save data
    :param data: input data (np.array, 2D)
    :param fname: path to file where data plot should be save
    :param logoffset: offset for log scale
    :param logscale: boolean if image should be displayed in log-scale
    :param cmap: colormap
    """
    assert data.ndim == 2
    path = os.path.dirname(fname) or '.'
    if not os.path.exists(path):
        os.makedirs(path)
            
    img = np.abs(data)
    if logscale:
        img /= np.max(img)
        img = np.log(img + logoffset)
    
    # save the image
    plt.imsave(fname, img, cmap=cmap, vmin=img.min(), vmax=img.max())

def normalize(img, vmin=None, vmax=None, max_int=255.0):
    """ normalize (magnitude) image
    :param image: input image (np.array)
    :param vmin: minimum input intensity value
    :param vmax: maximum input intensity value
    :param max_int: maximum output intensity value
    :return: normalized image
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())
    if vmin == None:
        vmin = np.min(img)
    if vmax == None:
        vmax = np.max(img)
    img = (img - vmin)*(max_int)/(vmax - vmin)
    img = np.minimum(max_int, np.maximum(0.0, img))
    return img

def phaseshow(img, title=''):
    """ Show phase image in grayscale
    :param img: input image (np.array)
    :param title: plot title
    """
    if np.iscomplexobj(img):
        print('[medutils.phaseshow] img is not complex!')
    img = np.angle(img)

    plt.figure()
    plt.imshow(img, cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.colorbar()
    plt.title(title)
    plt.set_cmap('hsv')

def contrastStretching(img, saturated_pixel=0.004):
    """ Contrast stretching according to imageJ
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
    :param img: input image (np.array)
    :param saturated_pixel: value to define the contrast stretching limits.
    :return: contrast stretched image
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    vmin = values[lim]
    vmax = values[-lim-1]
    img = (img - vmin)*(255.0)/(vmax - vmin)
    img = np.minimum(255.0, np.maximum(0.0, img))
    return img

def brighten(img, beta):
    """ Image brightening according to Matlab function brighten.
    :param img: input image (np.array)
    :param beta: value defining the brightening exponent.
    :return: brightened image
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())

    if np.max(img) > 1:
        img = img.copy() / 255.0

    assert beta > 0 and beta < 1
    tol = np.sqrt(2.2204e-16)
    gamma = 1 - min(1-tol, beta)
    img = img ** gamma
    return img

def getContrastStretchingLimits(img, saturated_pixel=0.004):
    """ Compute contrast stretching limits according to imageJ
        http://homepages.inf.ed.ac.uk/rbf/HIPR2/stretch.htm
    :param img: input image (np.array)
    :param saturated_pixel: value to define the contrast stretching limits.
    :return: contrast stretching lower and upper limit vmin, vmax.
    """
    if np.iscomplexobj(img):
        # print('img is complex! Take absolute value.')
        img = np.abs(img.copy())
    values = np.sort(img, axis=None)
    nr_pixels = np.size(values)
    lim = int(np.round(saturated_pixel*nr_pixels))
    vmin = values[lim]
    vmax = values[-lim-1]
    return vmin, vmax

def flip(img, axes=(0,1)):
    """ Flip image upside-down and left-right along specified axes
    :param img: input image (np.array)
    :param axes: tuple of axes over which the flipping is executed
    :return: flipped image
    """
    assert len(axes) == 2
    if img.ndim == 2 and axes==(0,1):
        return np.flipud(np.fliplr(img.copy()))
    elif img.ndim == 2 and axes != (0,1):
        raise ValueError("axes of 2d array have to equal (0,1)")
    else:
        axes = list(axes)
        full_axes = list(range(0, img.ndim))
        transpose_axes = [item for item in full_axes if item not in axes] + axes
        unwrap_axes = [transpose_axes.index(item) for item in full_axes]
        tmp_img = np.transpose(img.copy(), transpose_axes)
        tmp_shape = tmp_img.shape
        tmp_img = np.reshape(tmp_img, (np.prod(tmp_img.shape[:-2]),) + tmp_img.shape[-2:])
        for i in range(tmp_img.shape[0]):
            tmp_img[i] = np.flipud(np.fliplr(tmp_img[i]))
        tmp_img = np.reshape(tmp_img, tmp_shape)
        flipped_img = np.transpose(tmp_img, unwrap_axes)
        return flipped_img