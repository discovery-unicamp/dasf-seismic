#!/usr/bin/env python3

import dask.array as da
import numpy as np

try:
    import cupy as cp
    from glcm_cupy import Direction as glcm_direction
    from glcm_cupy import Features as glcm_features
    from glcm_cupy import glcm as glcm_gpu
except ImportError:
    pass

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:
    # XXX: Deprecated after release 0.19 of scikit-image
    from skimage.feature import greycomatrix as graycomatrix
    from skimage.feature import greycoprops as graycoprops

from dasf.transforms import Transform
from dasf.utils.types import is_dask_array, is_gpu_array
from skimage.feature import local_binary_pattern

from dasf_seismic.utils.utils import dask_overlap, dask_trim_internal


def get_glcm_gpu_feature(glcm_type):
    if glcm_type == "contrast":
        return glcm_features.CONTRAST
    elif glcm_type == "dissimilarity":
        return glcm_features.DISSIMILARITY
    elif glcm_type == "homogeneity":
        return glcm_features.HOMOGENEITY
    elif glcm_type == "asm" or glcm_type == "energy":
        return glcm_features.ASM
    elif glcm_type == "mean":
        return glcm_features.MEAN
    elif glcm_type == "correlation":
        return glcm_features.CORRELATION
    elif glcm_type == "variance" or glcm_type == "std":
        return glcm_features.VARIANCE
    elif glcm_type == "entropy":
        return glcm_features.ENTROPY
    else:
        raise NotImplementedError("GLCM type '%s' is not supported."
                                  % glcm_type)


def get_glcm_gpu_direction(direction):
    if direction == 0:
        return glcm_direction.EAST
    elif direction == np.pi/4:
        return glcm_direction.SOUTH_EAST
    elif direction == np.pi/2:
        return glcm_direction.SOUTH
    elif direction == 3*np.pi/2:
        return glcm_direction.SOUTH_WEST
    else:
        raise NotImplementedError("GLCM direction angle '%s' is not supported."
                                  % direction)


def get_glcm_cpu_feature(glcm_type):
    if glcm_type == "asm":
        return "ASM"
    return glcm_type


class GLCMGeneric(Transform):
    def __init__(self, glcm_type, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__()

        self._glcm_type = glcm_type
        self._levels = levels
        self._direction = direction
        self._distance = distance
        self._window = window
        self._glb_mi = glb_mi
        self._glb_ma = glb_ma

    def _operation_cpu(self, block, glcm_type_block, levels_block,
                       direction_block, distance_block, window, glb_mi, glb_ma,
                       pad=False):

        assert len(window) == 2
        kh, kw = np.array(window)//2
        new_atts = list()
        gl = ((block - glb_mi)/(glb_ma - glb_mi))*(levels_block - 1)
        gl = gl.astype(int)

        # Makes non-lazy execution match GPU non-lazy execution
        if pad:
            gl = np.pad(gl, pad_width=((0, 0), (kh, kh), (kw, kw)),
                        constant_values=0)

        d, h, w, = gl.shape
        for k in range(d):
            new_att = np.zeros((h, w), dtype=np.float32)
            gl_block = gl[k]
            for i in range(h):
                for j in range(w):
                    # Windows needs to fit completely in image
                    if i < kh or j < kw:
                        continue
                    if i > (h - kh - 1) or j > (w - kw - 1):
                        continue

                    # Calculate GLCM on a kh x kw window, default is 7x7
                    glcm_window = gl_block[i - kh:i + kh + 1,
                                           j - kw:j + kw + 1]
                    glcm = graycomatrix(glcm_window, [distance_block],
                                        [direction_block],
                                        levels=levels_block,
                                        symmetric=True, normed=True)

                    # Calculate property and replace center pixel
                    res = graycoprops(glcm, glcm_type_block)
                    new_att[i, j] = res[0, 0]

            new_atts.append(new_att.astype(block.dtype))
        result = np.asarray(new_atts, dtype=block.dtype)
        if pad:
            result = result[:, kh:-kh, kw:-kw]
        return result

    def _operation_gpu(self, block, glcm_type_block, levels_block,
                       direction_block, distance_block, window, glb_mi, glb_ma):
        assert len(window) == 2
        kh, kw = np.array(window)//2
        radius = max(kh, kw)
        step_size = distance_block
        padding = radius + step_size

        gl = ((block - glb_mi)/(glb_ma - glb_mi))*(levels_block - 1)
        gl = gl
        gl_pad = cp.pad(gl,
                        pad_width=((0, 0),
                                   (padding, padding),
                                   (padding, padding)),
                        constant_values=0
                        )
        image_batch = gl_pad[:, :, :, cp.newaxis]
        g = glcm_gpu(image_batch,
                     directions=[direction_block],
                     features=[glcm_type_block],
                     step_size=step_size,
                     radius=radius,
                     bin_from=levels_block,
                     bin_to=levels_block,
                     normalized_features=False,
                     skip_border=True,
                     verbose=False)
        return cp.asarray(g[..., glcm_type_block].squeeze(axis=3),
                          dtype=block.dtype)

    def _lazy_transform_gpu(self, X):
        mi = da.min(X) if self._glb_mi is None else self._glb_mi
        ma = da.max(X) if self._glb_ma is None else self._glb_ma

        # MIN and MAX are compute here, to avoid keeping the ALL data
        # in memory to compute MIN and MAX and then start computing the GLCM.
        # This countermeasure improves performance significantly. This also
        # computes MIN and MAX together, if necessary, and avoids MIN
        # recomputation.
        if is_dask_array(mi):
            mi = mi.compute()
        if is_gpu_array(mi):
            mi = mi.item()
        if is_dask_array(ma):
            ma = ma.compute()
        if is_gpu_array(ma):
            ma = ma.item()

        X_da = dask_overlap(X, kernel=(0, 7, 7), boundary=mi)

        glcm_type = get_glcm_gpu_feature(self._glcm_type)
        direction = get_glcm_gpu_direction(self._direction)

        X = X_da.map_blocks(self._operation_gpu, glcm_type,
                            self._levels, direction, self._distance,
                            self._window, mi, ma, dtype=X_da.dtype,
                            meta=cp.array((), dtype=X_da.dtype))

        X = dask_trim_internal(X, (0, 7, 7))
        if self._glcm_type in ["std", "energy"]:
            X = da.sqrt(X)

        return X

    def _lazy_transform_cpu(self, X):
        mi = da.min(X) if self._glb_mi is None else self._glb_mi
        ma = da.max(X) if self._glb_ma is None else self._glb_ma

        # This computes MIN and MAX together, if necessary, and avoids MIN
        # recomputation. It is almost impossible to have a cupy meta array,
        # but for sanity it is nice to check.
        if is_dask_array(mi):
            mi = mi.compute()
        if is_gpu_array(mi):
            mi = mi.item()
        if is_dask_array(ma):
            ma = ma.compute()
        if is_gpu_array(ma):
            ma = ma.item()

        X_da = dask_overlap(X, kernel=(0, 7, 7), boundary=mi)

        glcm_type = get_glcm_cpu_feature(self._glcm_type)

        X = X_da.map_blocks(self._operation_cpu, glcm_type,
                            self._levels, self._direction, self._distance,
                            self._window, mi, ma, dtype=X_da.dtype,
                            meta=np.array((), dtype=X_da.dtype))

        return dask_trim_internal(X, (0, 7, 7))

    def _transform_gpu(self, X):
        mi = cp.min(X) if self._glb_mi is None else self._glb_mi
        ma = cp.max(X) if self._glb_ma is None else self._glb_ma

        glcm_type = get_glcm_gpu_feature(self._glcm_type)
        direction = get_glcm_gpu_direction(self._direction)

        X = self._operation_gpu(X, glcm_type, self._levels,
                                direction, self._distance,
                                self._window, mi, ma)

        if self._glcm_type in ["std", "energy"]:
            X = cp.sqrt(X)

        return X

    def _transform_cpu(self, X):
        mi = np.min(X) if self._glb_mi is None else self._glb_mi
        ma = np.max(X) if self._glb_ma is None else self._glb_ma

        glcm_type = get_glcm_cpu_feature(self._glcm_type)

        return self._operation_cpu(X, glcm_type, self._levels,
                                   self._direction, self._distance,
                                   self._window, mi, ma, pad=True)


class GLCMContrast(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="contrast", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMDissimilarity(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="dissimilarity", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMASM(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="asm", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMMean(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="mean", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMCorrelation(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="correlation", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMHomogeneity(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="homogeneity", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMVariance(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="variance", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMEntropy(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="entropy", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMStandardDeviation(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="std", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class GLCMEnergy(GLCMGeneric):
    def __init__(self, levels=16, direction=np.pi/2, distance=1,
                 window=(7, 7), glb_mi=None, glb_ma=None):
        super().__init__(glcm_type="energy", levels=levels,
                         direction=direction, distance=distance,
                         window=window, glb_mi=glb_mi, glb_ma=glb_ma)


class LocalBinaryPattern2D(Transform):
    def __init__(self, radius=3, neighboors=8, method='default'):
        super().__init__()

        self._radius = radius
        self._neighboors = neighboors * radius
        self._method = method

    def _operation(self, block, neighboors, radius, method, block_info=None):
        sub_cube = list()
        for i in range(0, block.shape[0]):
            sub_cube.append(local_binary_pattern(block[i, :, :],
                                                 neighboors,
                                                 radius, method))
        return np.asarray(sub_cube, dtype=block.dtype)

    def _lazy_transform_cpu(self, X):
        kernel = (1, min(int(X.shape[1]/4), 1000), int(X.shape[2]))
        axes = (0, 2*self._radius, 2*self._radius)

        X_da = dask_overlap(X, kernel, axes=axes,
                            boundary='periodic')

        result = X_da.map_blocks(self._operation, self._neighboors,
                                 self._radius, self._method,
                                 dtype=X_da.dtype,
                                 meta=np.array((), dtype=X_da.dtype))

        result = dask_trim_internal(result, kernel, axes)

        return result

    def _lazy_transform_gpu(self, X):
        raise NotImplementedError("CuCIM does not have any LBP method "
                                  "implemented yet")

    def _transform_cpu(self, X):
        # Padding to get same result as lazy implementation in border region
        r = self._radius
        X = np.pad(X, pad_width=((0, 0), (r, r), (r, r)), mode="wrap")
        result = self._operation(X, self._neighboors,
                                 self._radius, self._method)

        result = result[:, r:-r, r:-r]
        return result

    def _transform_gpu(self, X):
        raise NotImplementedError("CuCIM does not have any LBP method "
                                  "implemented yet")


class LocalBinaryPattern3D(Transform):
    def __init__(self, method="diagonal", use_unique=True):
        super().__init__()

        self._method = method
        self._use_unique = use_unique

    def _operation_unique(self, block, unique_array):
        for i in range(len(unique_array)):
            block[block == unique_array[i]] = i

        return block

    def _operation_cpu(self, block):
        img_lbp = np.zeros_like(block)
        neighboor = 3
        s0 = int(neighboor/2)
        for ih in range(0, block.shape[0] - neighboor + s0):
            for iw in range(0, block.shape[1] - neighboor + s0):
                for iz in range(0, block.shape[2] - neighboor + s0):
                    img = block[ih:ih + neighboor,
                                iw:iw + neighboor,
                                iz:iz + neighboor]
                    center = img[1, 1, 1]
                    img_aux_vector = img.flatten()

                    # Delete centroids
                    del_vec = [1, 3, 4, 5, 7, 9, 10, 11, 12, 13,
                               14, 15, 16, 17, 19, 21, 22, 23, 25]

                    img_aux_vector = np.delete(img_aux_vector, del_vec)

                    weights = 2 ** np.arange(len(img_aux_vector),
                                             dtype=np.uint64)

                    mask_vec = np.zeros(len(img_aux_vector), dtype=np.int8)

                    idx_max = img_aux_vector.argmax()
                    idx_min = img_aux_vector.argmin()

                    if img_aux_vector[idx_max] > center:
                        mask_vec[idx_max] = 1

                    if img_aux_vector[idx_min] < center:
                        mask_vec[idx_min] = 1

                    num = np.sum(weights * mask_vec)

                    img_lbp[ih + 1, iw + 1, iz + 1] = num
        return img_lbp

    def _operation_gpu(self, block):
        __lbp_gpu = cp.RawKernel(r'''
            extern "C" __global__
            void local_binary_pattern_gpu(const float *a, float *out,
                                          float max_local, float min_local,
                                          unsigned int nx, unsigned int ny,
                                          unsigned int nz) {
                unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
                unsigned int idy = threadIdx.y + blockIdx.y * blockDim.y;
                unsigned int idz = threadIdx.z + blockIdx.z * blockDim.z;
                int i, j, k;
                float max, min;
                unsigned int center, index, kernel_idx;
                unsigned int max_idx, min_idx;
                float exp, sum, mult, n;
                unsigned char kernel[27] = {
                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                   0, 0, 0, 0, 0, 0, 0, 0, 0,
                };
                max = min_local - 1;
                min = max_local + 1;
                if ((idx > 0 && idy > 0 && idz > 0) &&
                    (idx < nx - 1) && (idy < ny - 1) && (idz < nz - 1)) {
                    center = ((ny * nz) * idx) + (idy * nz + idz);
                    for(i = -1; i <= 1; i = i + 2) {
                        for(j = -1; j <= 1; j = j + 2) {
                            for(k = -1; k <= 1; k = k + 2) {
                                /* Avoid illegal memory access */
                                if ((idx + i >= nx ||
                                     idy + j >= ny ||
                                     idz + k >= nz) ||
                                    (idx + i < 0 ||
                                     idy + j < 0 ||
                                     idz + k < 0)) {
                                    continue;
                                }
                                index = (((ny * nz) * (idx + i)) +
                                         ((idy + j) * nz + (idz + k)));
                                kernel_idx = ((9 * (i + 1)) + ((j + 1) *
                                               3 + (k + 1)));
                                if (max < a[index]) {
                                    if (a[center] < a[index]) {
                                        max = a[index];
                                        max_idx = kernel_idx;
                                    }
                                }
                                if (min > a[index]) {
                                    if (a[center] > a[index]) {
                                        min = a[index];
                                        min_idx = kernel_idx;
                                    }
                                }
                            }
                        }
                    }
                    if (max < max_local + 1 && max > min_local - 1) {
                        kernel[max_idx] = 1;
                    }
                    if (min > min_local - 1 && min < max_local + 1) {
                        kernel[min_idx] = 1;
                    }
                    mult = exp = sum = 0;
                    for(i = 0; i <= 2; i = i + 2) {
                        for(j = 0; j <= 2; j = j + 2) {
                            for(k = 0; k <= 2; k = k + 2) {
                                if (kernel[(9 * i) + (j * 3 + k)] == 1) {
                                    // Implementing our own pow() function
                                    n = 0;
                                    mult = 1;
                                    while(n < exp) {
                                        mult *= 2;
                                        n++;
                                    }
                                    sum += mult;
                                }
                                exp++;
                            }
                        }
                    }
                    out[center] = sum;
                }
            }
        ''', 'local_binary_pattern_gpu')

        dimx = block.shape[0]
        dimy = block.shape[1]
        dimz = block.shape[2]

        out = cp.zeros((dimz * dimy * dimx), dtype=cp.float32)
        inp = cp.asarray(block, dtype=cp.float32)

        # Numpy is faster than Cupy for min and max
        min_local = np.min(block.flatten()).get()
        max_local = np.max(block.flatten()).get()

        block_size = 10

        cuda_grid = (int(np.ceil(dimx/block_size)),
                     int(np.ceil(dimy/block_size)),
                     int(np.ceil(dimz/block_size)),)
        cuda_block = (block_size, block_size, block_size,)

        __lbp_gpu(cuda_grid, cuda_block, (inp, out, cp.float32(max_local),
                                          cp.float32(min_local),
                                          cp.int32(dimx),
                                          cp.int32(dimy), cp.int32(dimz)))

        return cp.asarray(out).reshape(dimx, dimy, dimz).astype(block.dtype)

    def _lazy_transform_cpu(self, X):
        kernel = (2, 2, 2)

        X_da = dask_overlap(X, kernel=kernel,
                            boundary='periodic')

        result = X_da.map_blocks(self._operation_cpu,
                                 dtype=X_da.dtype,
                                 meta=np.array((), dtype=X_da.dtype))

        result = dask_trim_internal(result, kernel)

        if self._use_unique:
            unique = da.unique(result)

            result = result.map_blocks(self._operation_unique, unique,
                                       dtype=result.dtype,
                                       meta=np.array((), dtype=X_da.dtype))

        return result

    def _lazy_transform_gpu(self, X):
        kernel = (2, 2, 2)

        X_da = dask_overlap(X, kernel=kernel,
                            boundary='periodic')

        result = X_da.map_blocks(self._operation_gpu,
                                 dtype=X_da.dtype,
                                 meta=cp.array((), dtype=X_da.dtype))

        result = dask_trim_internal(result, kernel)

        if self._use_unique:
            unique = da.unique(result)

            result = result.map_blocks(self._operation_unique, unique,
                                       dtype=result.dtype,
                                       meta=cp.array((), dtype=X_da.dtype))

        return result

    def _transform_gpu(self, X):
        # Padding to get same result as lazy implementation in border region
        X = cp.pad(X, pad_width=2, mode="wrap")
        result = self._operation_gpu(X)

        trim = tuple([slice(2, -2) for _ in X.shape])
        result = result[trim]

        if self._use_unique:
            unique = cp.unique(result)

            result = self._operation_unique(result, unique)

        return result

    def _transform_cpu(self, X):
        # Padding to get same result as lazy implementation in border region
        X = np.pad(X, pad_width=2, mode="wrap")
        result = self._operation_cpu(X)

        trim = tuple([slice(2, -2) for _ in X.shape])
        result = result[trim]

        if self._use_unique:
            unique = np.unique(result)

            result = self._operation_unique(result, unique)

        return result
