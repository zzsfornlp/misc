#

import numpy as np
from src2.nnlight import BK, Inits, Embed, BiaffineScorer, LossNeg

class model2:
    def __init__(self, args, dataset, data_iterator):
        self.args = args
        self.dataset = dataset
        self.data_iterator = data_iterator
        #
        # options
        self.emb_dim = args.emb_dim
        self.subword = dataset.subword
        self.num_neg = args.neg
        #
        self.separate_wc = args.separate_wc
        self.real_num_label = self.dataset.label_vocab_size
        self.real_num_word = self.dataset.vocab_size
        # loss function
        # fix NEG-loss
        # warmup
        self.warmup_percent = args.warmup_percent
        # progress check
        self.last_progress_check = 0.
        #
        if self.separate_wc:
            if self.subword:
                self.vocab_size = self.real_num_word + self.dataset.word_vocab_size
            else:
                self.vocab_size = self.real_num_word * 2
            self.num_label = self.real_num_label * 2
        else:
            self.vocab_size = self.real_num_word
            self.num_label = self.real_num_label
        # graident clipping
        # TODO

        # components
        self.scorer = BiaffineScorer(self.num_neg)
        self.losser = LossNeg(self.num_neg)
        # parameters
        # Label-W
        label_init = Inits.eyes(self.emb_dim, self.num_label).reshape([self.num_label, -1])
        self.label_embs = Embed(self.num_label, self.emb_dim*self.emb_dim, init=label_init, explaining_dims=[self.emb_dim, self.emb_dim])
        # Word-W
        if self.subword:
            self.word_embs = Embed(self.vocab_size, self.emb_dim, subword_mode=True)
            self.subword_pad_idx = dataset.PAD_IDX
            self._zero_paddings()
        else:
            self.word_embs = Embed(self.vocab_size, self.emb_dim)
        # =====
        # zero ctx embeddings if separate_wc
        if self.separate_wc and self.vocab_size > self.real_num_word:
            self.word_embs.E[self.real_num_word:] = 0.
        # =====
        #
        self.cur_lr = self.init_lr = args.init_lr
        # TODO
        self.freeze_Label_M = False

    def _zero_paddings(self):
        self.word_embs.E[self.subword_pad_idx] = 0.

    def is_warming_up(self):
        return self.data_iterator.progress < self.warmup_percent or self.freeze_Label_M

    # inputs are all flattened
    def step(self, datum, training_stats, return_loss):
        # =====
        # check warmup
        cur_progress = self.data_iterator.progress
        if cur_progress - self.last_progress_check >= 0.0002:
            self.last_progress_check = cur_progress
            print(f"Check-progress: {cur_progress}, is_warm={self.is_warming_up()}")
        # =====
        bsize = len(datum[4])
        is_warming_up = self.is_warming_up()
        if is_warming_up:
            emb_lr_mul = self.args.emb_lr_factor_before
        else:
            emb_lr_mul = self.args.emb_lr_factor_after
        # prepare data (add to the second area for ctx if separate, labels are already handled in data-loader!)
        datum = [None if z is None else BK.as_tensor(z) for z in datum]
        # add H word idxes here by GPU
        if self.separate_wc:
            datum[0] += self.real_num_word
            datum[2] += self.real_num_word
        # =====
        # forward
        pos_h, pos_m, neg_h, neg_m, dep_labels, ph_counts, pm_counts, nh_counts, nm_counts, ph_q, pm_q, nh_q, nm_q = datum
        # Embedding
        ishape_pos = [bsize, ]
        ishape_neg = [bsize, self.num_neg]
        if self.subword:
            ishape_pos.append(np.prod(pos_h.shape)//np.prod(ishape_pos))
            ishape_neg.append(np.prod(neg_h.shape)//np.prod(ishape_neg))
        #
        label_embs = self.label_embs.forward(dep_labels, None, [bsize])
        pos_h_embs = self.word_embs.forward(pos_h, ph_counts, ishape_pos)
        neg_h_embs = self.word_embs.forward(neg_h, nh_counts, ishape_neg)
        pos_m_embs = self.word_embs.forward(pos_m, pm_counts, ishape_pos)
        neg_m_embs = None if self.separate_wc else self.word_embs.forward(neg_m, nm_counts, ishape_neg)
        embs_pack = (label_embs, pos_h_embs, neg_h_embs, pos_m_embs, neg_m_embs)
        # Scoring
        scores_pack = self.scorer.forward(embs_pack)
        # Loss
        probs_pack, loss_pack = self.losser.forward(scores_pack)
        # =====
        # backward
        # Loss
        g_scores_pack = self.losser.backward(probs_pack, bsize)
        # Scoring
        g_embs_pack = self.scorer.backward(embs_pack, scores_pack, g_scores_pack)
        # Embeddings
        g_label_embs, g_pos_h_embs, g_neg_h_embs, g_pos_m_embs, g_neg_m_embs = g_embs_pack
        if not is_warming_up:
            self.label_embs.backward(g_label_embs, dep_labels, None, [bsize])
        self.word_embs.backward(g_pos_h_embs, pos_h, ph_counts, ishape_pos)
        self.word_embs.backward(g_neg_h_embs, neg_h, nh_counts, ishape_neg)
        self.word_embs.backward(g_pos_m_embs, pos_m, pm_counts, ishape_pos)
        if not self.separate_wc:
            self.word_embs.backward(g_neg_m_embs, neg_m, nm_counts, ishape_neg)
        # =====
        # record and set lr
        if return_loss:
            pos_h_loss, pos_m_loss, neg_h_loss, neg_m_loss, loss = loss_pack
            training_stats["pos_h_loss"].append(BK.as_numpy(pos_h_loss).item())
            training_stats["pos_m_loss"].append(BK.as_numpy(pos_m_loss).item())
            training_stats["neg_h_loss"].append(BK.as_numpy(neg_h_loss).item())
            training_stats["neg_m_loss"].append(BK.as_numpy(neg_m_loss).item())
            training_stats["loss"].append(BK.as_numpy(loss).item())
        if self.data_iterator.tot_batches % self.args.lr_update_rate == 0:
            self.cur_lr = self.init_lr * (1 - self.data_iterator.progress)
        # simple SGD update
        if not is_warming_up:
            self.label_embs.update(self.cur_lr)
        # TODO
        # self.word_embs.update(self.cur_lr*emb_lr_mul)
        # =====
        # clear PAD-IDX
        if self.subword:
            self._zero_paddings()
        # re-write fix embeddings
        # TODO
