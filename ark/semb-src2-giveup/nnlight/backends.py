#

# the backends PyTorch, numpy/scipy or cupy

import numpy as np
import scipy
from scipy.sparse import coo_matrix
import torch
import cupy as cp
from .utils import zlog

_SEED = 12345

class BackendPyTorch:
    pass

class BackendNumpy:
    @staticmethod
    def init():
        np.random.seed(_SEED)

    as_tensor = np.asarray
    index_select = lambda a, axis, inds: np.take(a, inds, axis)
    reshape = np.reshape
    sum = np.sum
    unsqueeze = np.expand_dims
    tile = np.tile
    concat = np.concatenate
    matmul = np.matmul
    squeeze = np.squeeze
    swapaxes = np.swapaxes
    log = np.log
    mean = np.mean
    sigmoid = lambda x: 1. / (1. + np.exp(-x))

    zero_scalar = lambda: np.array(0., dtype=np.float32)
    as_numpy = np.asarray

    @staticmethod
    def hybrid_sparse_2d(index_all, grads_all, shape):
        num_row = index_all.shape[0]
        emb_size = shape[-1]
        # (data, (row, col))
        x = scipy.sparse.coo_matrix(
            (grads_all.reshape(-1), (np.repeat(index_all, emb_size), np.tile(np.arange(emb_size), num_row))),
            shape=shape, dtype=np.float32)
        return x

class BackendCupy:
    @staticmethod
    def init():
        cp.random.seed(_SEED)

    as_tensor = cp.asarray
    index_select = lambda a, axis, inds: cp.take(a, inds, axis)
    reshape = cp.reshape
    sum = cp.sum
    unsqueeze = cp.expand_dims
    tile = cp.tile
    concat = cp.concatenate
    matmul = cp.matmul
    squeeze = cp.squeeze
    swapaxes = cp.swapaxes
    log = cp.log
    mean = cp.mean
    sigmoid = lambda x: 1./(1.+cp.exp(-x))

    zero_scalar = lambda : cp.array(0., dtype=cp.float32)
    as_numpy = cp.asnumpy

    @staticmethod
    def hybrid_sparse_2d(index_all, grads_all, shape):
        num_row = index_all.shape[0]
        emb_size = shape[-1]
        # (data, (row, col))
        x = cp.sparse.coo_matrix(
            (grads_all.reshape(-1), (cp.repeat(index_all, emb_size), cp.tile(cp.arange(emb_size), num_row))),
            shape=shape, dtype=cp.float32)
        return x

# =====
# select by env-name
import os
BK_NAME = str.lower(os.environ.get("BK", "torch"))
if BK_NAME in {"torch", "pytorch"}:
    BK = BackendPyTorch
    zlog("Using backend of PyTorch")
elif BK_NAME in {"numpy", "np"}:
    BK = BackendNumpy
    zlog("Using backend of NumPy(CPU)")
elif BK_NAME in {"cupy", "cp"}:
    BK = BackendCupy
    zlog("Using backend of CuPy(GPU)")
else:
    raise NotImplementedError(f"Unk BK {BK_NAME}")
#

# overall initialization
BK.init()
