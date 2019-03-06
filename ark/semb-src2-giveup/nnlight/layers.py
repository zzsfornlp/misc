#

# common used layers with their forwards/backwards

from typing import List
import numpy as np
from .backends import BK

# helpers for init
class Inits:
    zeros = np.zeros

    @staticmethod
    def eyes(size: int, batch_size: int=None, dtype=np.float32):
        if batch_size is None:
            return np.eye(size).astype(dtype)
        else:
            arrs = [np.eye(size) for _ in range(batch_size)]
            ret = np.stack(arrs).astype(dtype)
            return ret

    @staticmethod
    def uniform(shape, rr: float=1., dtype=np.float32):
        arr = np.random.uniform(-rr, rr, size=shape).astype(dtype)
        return arr

    @staticmethod
    def glorot(shape, factor: float=1., lookup=False, dtype=np.float32):
        eff_shape = (shape[-1], ) if lookup else shape
        rr = factor * np.sqrt(3.0*len(eff_shape)/(sum(eff_shape)))
        return Inits.uniform(shape, rr, dtype)

# embedding layer
class Embed:
    def __init__(self, num: int, dim: int, subword_mode: bool=False, pad_idx: int=None, init: np.ndarray=None, explaining_dims=None):
        self.num = num
        self.dim = dim
        self.shape = (num, dim)
        self.subword_mode = subword_mode
        self.pad_idx = pad_idx
        if explaining_dims is None:
            self.explaining_dims = [dim]
        else:
            assert np.prod(explaining_dims) == dim
            self.explaining_dims = list(explaining_dims)
        # parameters
        if init is None:
            init = Inits.glorot(self.shape, lookup=True)
        else:
            assert self.shape == init.shape
        self.E = BK.as_tensor(init)
        #
        self.clear()

    # for efficiency, the input indexes are all flattened!
    # indexes_t: prod(*, subword?), counts_t-for-subword: *, indexes_shape: [bs, neg?, sub-size?]
    def forward(self, indexes_t, counts_t, indexes_shape: List[int]):
        DIM = self.dim
        emb_flat = BK.index_select(self.E, 0, indexes_t)        # [*, dim]
        if self.subword_mode:
            subword_size = indexes_shape[-1]
            emb_sub = BK.reshape(emb_flat, [-1, subword_size, DIM])    # [*, sub, dim]
            emb_avg = BK.sum(emb_sub, 1) / BK.unsqueeze(counts_t, 1)    # [*, dim]
            final_shape = indexes_shape[:-1] + self.explaining_dims
            return BK.reshape(emb_avg, final_shape)
        else:
            final_shape = indexes_shape + self.explaining_dims
            return BK.reshape(emb_flat, final_shape)

    #
    def backward(self, g_embs, indexes_t, counts_t, indexes_shape: List[int]):
        DIM = self.dim
        flatten_shape = [-1, DIM]
        g_embs_shaped = BK.reshape(g_embs, flatten_shape)                   # [*, dim]
        if self.subword_mode:
            g_embs_div = g_embs_shaped / BK.unsqueeze(counts_t, 1)          # [*, dim]
            # expand subword dimension
            subword_size = indexes_shape[-1]
            g_embs_tile = BK.tile(BK.unsqueeze(g_embs_div, 1), [1,subword_size,1])      # [*, sub, dim]
            g_embs_shaped = BK.reshape(g_embs_tile, flatten_shape)          # [*, dim]
        self.indexes.append(indexes_t)              # [*]
        self.gradients.append(g_embs_shaped)        # [*, dim]

    def clear(self):
        self.indexes = []       # list of [*(int)]
        self.gradients = []     # list of [*, DIM]

    def update(self, lrate):
        #
        index_all = BK.concat(self.indexes, 0)
        grads_all = BK.concat(self.gradients, 0)
        # get rid of paddding
        if self.pad_idx is not None:
            # make use of nonzero
            raise NotImplementedError("TODO")
        # update with sparse matrix
        g_embs = BK.hybrid_sparse_2d(index_all, grads_all, self.shape)
        # TODO: clip!
        # TODO: strangely, this one triggers Segment Fault with cupy?
        # TODO: oh, no, is it doing to_dense()?
        self.E -= lrate * g_embs
        # TODO: and with numpy&scipy, this turns to be a matrix
        self.E = BK.as_tensor(self.E)
        # clear gradient
        self.clear()

#
class BiaffineScorer:
    def __init__(self, NEG):
        self.neg_score_shape = [-1, NEG]
        self.neg_g_shape1 = [-1, NEG, 1]
        self.neg_g_shape2 = [-1, 1, NEG]
        # self.transpose_axis = [1, 2]

    def forward(self, embs_pack):
        # [bs, Dh, Dm], [bs, Dh], [bs, neg, Dh], [bs, Dm], [bs, neg, Dm]
        label_embs, pos_h_embs, neg_h_embs, pos_m_embs, neg_m_embs = embs_pack
        # =====
        # matrix multiplications to get the bilinear scores
        # step1: pos_h * label: [bs, 1, Dh] * [bs, Dh, Dm] -> [bs, 1, Dm] -> [bs, D']
        pos_h_mul_label = BK.matmul(BK.unsqueeze(pos_h_embs, 1), label_embs)
        pos_h_mul_label = BK.squeeze(pos_h_mul_label, 1)
        # step2: neg_h * label: [bs, N, Dh] * [bs, Dh, Dm] -> [bs, N, Dm]
        neg_h_mul_label = BK.matmul(neg_h_embs, label_embs)
        # step3: pos_h_mul * pos_m: [bs, Dh] . [bs, Dm] sum -> [bs]
        pos_score = BK.sum(pos_h_mul_label * pos_m_embs, 1)
        # step4: neg_h_mul * pos_m: [bs, N, Dh] * [bs, Dm, 1] -> [bs, N, 1] -> [bs, N]
        neg_h_score = BK.matmul(neg_h_mul_label, BK.unsqueeze(pos_m_embs, 2))
        neg_h_score = BK.reshape(neg_h_score, self.neg_score_shape)
        # step5: neg_m * pos_h_mul: [bs, N, Dm] * [bs, Dh, 1] -> [bs, N, 1] -> [bs, N]
        if neg_m_embs is None:
            neg_m_score = None
        else:
            neg_m_score = BK.matmul(neg_m_embs, BK.unsqueeze(pos_h_mul_label, 2))
            neg_m_score = BK.reshape(neg_m_score, self.neg_score_shape)
        # =====
        # return output scores and hidden layers
        return pos_score, neg_h_score, neg_m_score, pos_h_mul_label, neg_h_mul_label

    #
    def backward(self, embs_pack, scores_pack, g_scores_pack):
        label_embs, pos_h_embs, neg_h_embs, pos_m_embs, neg_m_embs = embs_pack
        pos_score, neg_h_score, neg_m_score, pos_h_mul_label, neg_h_mul_label = scores_pack
        g_pos_score, g_neg_h_score, g_neg_m_score = g_scores_pack
        #
        no_neg_m = (neg_m_embs is None)
        # =====
        # gradient to embeddings: back through the 5 matrix multiplications
        # g_pos_h_embs = 0.
        # g_pos_m_embs = 0.
        # g_neg_h_embs = 0.
        # g_neg_m_embs = 0.
        # g_label_embs = 0.
        # g_pos_h_mul_label = 0.
        # g_neg_h_mul_label = 0.
        # step5:
        if no_neg_m:
            g_neg_m_embs = None
            g_pos_h_mul_label = None
        else:
            # [bs, N, 1] * [bs, 1, D] -> [bs, N, D]
            g_neg_m_embs = BK.matmul(BK.reshape(g_neg_m_score, self.neg_g_shape1), BK.unsqueeze(pos_h_mul_label, 1))
            # [bs, 1, N] * [bs, N, D] -> [bs, 1, D] -> [bs, D]
            g_pos_h_mul_label = BK.matmul(BK.reshape(g_neg_m_score, self.neg_g_shape2), neg_m_embs)
            g_pos_h_mul_label = BK.squeeze(g_pos_h_mul_label, 1)
        # step4: similar to step5
        # [bs, N, 1] * [bs, 1, D] -> [bs, N, D]
        g_neg_h_mul_label = BK.matmul(BK.reshape(g_neg_h_score, self.neg_g_shape1), BK.unsqueeze(pos_m_embs, 1))
        # [bs, 1, N] * [bs, N, D] -> [bs, 1, D] -> [bs, D]
        g_pos_m_embs = BK.matmul(BK.reshape(g_neg_h_score, self.neg_g_shape2), neg_h_mul_label)
        g_pos_m_embs = BK.squeeze(g_pos_m_embs, 1)
        # step3:
        # [bs] . [bs, D] -> [bs, D]
        if no_neg_m:
            g_pos_h_mul_label = BK.unsqueeze(g_pos_score, 1) * pos_m_embs
        else:
            g_pos_h_mul_label += (BK.unsqueeze(g_pos_score, 1) * pos_m_embs)
        g_pos_m_embs += (BK.unsqueeze(g_pos_score, 1) * pos_h_mul_label)
        # step2: need to transpose here
        # transpose_axis = self.transpose_axis
        # [bs, N, D'] * [bs, D, D'].t() -> [bs, N, D]
        g_neg_h_embs = BK.matmul(g_neg_h_mul_label, BK.swapaxes(label_embs, 1,2))
        # [bs, N, D].t() * [bs, N, D'] -> [bs, D, D']
        g_label_embs = BK.matmul(BK.swapaxes(neg_h_embs, 1,2), g_neg_h_mul_label)
        # step1:
        # [bs, 1, D'] * [bs, D, D'].t() -> [bs, 1, D] -> [bs, D]
        g_pos_h_embs = BK.matmul(BK.unsqueeze(g_pos_h_mul_label, 1), BK.swapaxes(label_embs, 1,2))
        g_pos_h_embs = BK.squeeze(g_pos_h_embs, 1)
        # [bs, 1, D].t() * [bs, 1, D'] -> [bs, D, D']
        g_label_embs += (BK.matmul(BK.unsqueeze(pos_h_embs, 2), BK.unsqueeze(g_pos_h_mul_label, 1)))
        # =====
        return g_label_embs, g_pos_h_embs, g_neg_h_embs, g_pos_m_embs, g_neg_m_embs

# neg-sampling loss (simple sum of log binary loss)
class LossNeg:
    def __init__(self, NEG):
        self.NEG = NEG

    def forward(self, scores_pack):
        pos_score, neg_h_score, neg_m_score, _, _ = scores_pack
        # =====
        # probs
        prob_pos_h = BK.sigmoid(pos_score)  # [bs]
        prob_neg_h = BK.sigmoid(-neg_h_score)  # [bs, neg]
        # losses
        pos_h_loss = BK.mean(-BK.log(prob_pos_h))
        neg_h_loss = BK.mean(-BK.log(prob_neg_h)) * self.NEG
        # =====
        if neg_m_score is None:
            prob_pos_m = prob_neg_m = None
            pos_m_loss = neg_m_loss = BK.zero_scalar()
            loss = pos_h_loss + neg_h_loss
        else:
            prob_pos_m = prob_pos_h
            prob_neg_m = BK.sigmoid(-neg_m_score)
            pos_m_loss = BK.mean(-BK.log(prob_pos_m))
            neg_m_loss = BK.mean(-BK.log(prob_neg_m)) * self.NEG
            loss = pos_h_loss + pos_m_loss + neg_h_loss + neg_m_loss
        return (prob_pos_h, prob_pos_m, prob_neg_h, prob_neg_m), (pos_h_loss, pos_m_loss, neg_h_loss, neg_m_loss, loss)

    def backward(self, probs_pack, bsize):
        prob_pos_h, prob_pos_m, prob_neg_h, prob_neg_m = probs_pack
        #
        if prob_neg_m is None:
            g_pos_score = (prob_pos_h - 1.) / bsize
            g_neg_h_score = (1. - prob_neg_h) / bsize
            g_neg_m_score = None
        else:
            g_pos_score = (prob_pos_h + prob_pos_m - 2.) / bsize
            g_neg_h_score = (1. - prob_neg_h) / bsize
            g_neg_m_score = (1. - prob_neg_m) / bsize
        return g_pos_score, g_neg_h_score, g_neg_m_score
