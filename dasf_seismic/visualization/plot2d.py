#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from dasf.utils.types import is_dask_array, is_gpu_array
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_slice(dslice, title=None, cmap='gray', img_scale=0.015,
               interpolation='bicubic', is_discrete=False, filename=None):
    img_h = int(dslice.shape[0] * img_scale)
    img_w = int(dslice.shape[1] * img_scale)

    plt.figure(figsize=(img_w, img_h))
    if title:
        plt.title(title)
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.1)

    if is_discrete:
        cmap = plt.get_cmap(cmap, np.max(dslice)-np.min(dslice)+1)
        im = ax.imshow(dslice, cmap=cmap, interpolation=None,
                       vmin=np.min(dslice)-.5, vmax=np.max(dslice)+.5)
        plt.colorbar(im, cax=cax, ticks=np.arange(np.min(dslice),
                                                  np.max(dslice)+1))
    else:
        im = ax.imshow(dslice, cmap=cmap, interpolation=interpolation)
        plt.colorbar(im, cax=cax)

    if filename:
        plt.savefig(fname=filename)


class Plot2DIline:
    def __init__(self, name, iline_index=-1, title=None, cmap='gray',
                 img_scale=0.015, interpolation='bicubic', is_discrete=False,
                 filename=None, swapaxes=None):
        self.iline_index = iline_index
        self.title = title
        self.cmap = cmap
        self.img_scale = img_scale
        self.interpolation = interpolation
        self.is_discrete = is_discrete
        self.filename = filename
        self.swapaxes = swapaxes

    def plot(self, X):
        if self.iline_index >= 0:
            if is_dask_array(X):
                new_data = X[self.iline_index, :, :].compute()
            else:
                new_data = X[self.iline_index, :, :]
        else:
            if is_dask_array(X):
                new_data = X.compute()
            else:
                new_data = X

        if is_gpu_array(new_data):
            new_data = new_data.get()

        if self.swapaxes is None:
            pass
        elif len(self.swapaxes) == 2:
            new_data = np.swapaxes(new_data, self.swapaxes[0], self.swapaxes[1])
        else:
            raise Exception("Attribute swapaxes is not well defined. "
                            "It requires a 2-D tuple or array.")

        plot_slice(new_data, self.title, self.cmap, self.img_scale,
                   self.interpolation, self.is_discrete, self.filename)

        return new_data


class Plot2DXline:
    def __init__(self, name, xline_index=-1, title=None, cmap='gray',
                 img_scale=0.015, interpolation='bicubic', is_discrete=False,
                 filename=None, swapaxes=None):
        self.xline_index = xline_index
        self.title = title
        self.cmap = cmap
        self.img_scale = img_scale
        self.interpolation = interpolation
        self.is_discrete = is_discrete
        self.filename = filename
        self.swapaxes = swapaxes

    def plot(self, X):
        if self.xline_index >= 0:
            if is_dask_array(X):
                new_data = X[:, self.xline_index, :].compute()
            else:
                new_data = X[:, self.xline_index, :]
        else:
            if is_dask_array(X):
                new_data = X.compute()
            else:
                new_data = X

        if is_gpu_array(new_data):
            new_data = new_data.get()

        if self.swapaxes is None:
            pass
        elif len(self.swapaxes) == 2:
            new_data = np.swapaxes(new_data, self.swapaxes[0], self.swapaxes[1])

        plot_slice(new_data, self.title, self.cmap, self.img_scale,
                   self.interpolation, self.is_discrete, self.filename)

        return new_data


class Plot2DDepth:
    def __init__(self, name, depth=-1, title=None, cmap='gray',
                 img_scale=0.015, interpolation='bicubic', is_discrete=False,
                 filename=None, swapaxes=None):
        self.depth = depth
        self.title = title
        self.cmap = cmap
        self.img_scale = img_scale
        self.interpolation = interpolation
        self.is_discrete = is_discrete
        self.filename = filename
        self.swapaxes = swapaxes

    def plot(self, X):
        if self.depth >= 0:
            if is_dask_array(X):
                new_data = X[:, :, self.depth].compute()
            else:
                new_data = X[:, :, self.depth]
        else:
            if is_dask_array(X):
                new_data = X.compute()
            else:
                new_data = X

        if is_gpu_array(new_data):
            new_data = new_data.get()

        if self.swapaxes is None:
            pass
        elif len(self.swapaxes) == 2:
            new_data = np.swapaxes(new_data, self.swapaxes[0], self.swapaxes[1])
        else:
            raise Exception("Attribute swapaxes is not well defined. "
                            "It requires a 2-D tuple or array.")

        plot_slice(new_data, self.title, self.cmap, self.img_scale,
                   self.interpolation, self.is_discrete, self.filename)

        return new_data
