#

# helper functions with tensorflow

import tensorflow as tf
import numpy as np
from typing import Sequence
from .utils import my_print

# =====
# general helpers

# mode
class TFMode:
    TRAIN = 'train'
    EVAL = 'eval'
    INFER = 'infer'

# activations
class TFTables:
    TF_ACTIVATIONS = {
        "tanh": tf.nn.tanh, "linear": lambda x, **kwargs: x, "softmax": tf.nn.softmax, "relu": tf.nn.relu
    }
    TF_POOLINGS = {
        # -2 means the one up embed layer
        "max": lambda x, **kwargs: tf.reduce_max(x, axis=-2, **kwargs),
        "avg": lambda x, **kwargs: tf.reduce_mean(x, axis=-2, **kwargs),
    }

def init_seed(seed=1234):
    tf.random.set_random_seed(seed)
    np.random.seed(seed*2+1)

def get_shape(t, idx=None):
    shape = t.shape
    if idx is None:
        return [int(z) for z in t.shape]
    else:
        return int(shape[idx])

# activation and/or dropout
def act_and_drop(val, output_act=None, pooling=None, output_drop=0., drop_fix_dim=None):
    # act, pooling, dropout
    if output_act is not None:
        val = TFTables.TF_ACTIVATIONS[output_act](val, name="val_act")
    if pooling is not None:
        val = TFTables.TF_POOLINGS[pooling](val, name="val_pool")
    if output_drop > 0.:
        if drop_fix_dim is None:
            noise_shape = None
        else:
            noise_shape = get_shape(val)
            noise_shape[drop_fix_dim] = 1
        # todo(warn): construct different models for train/eval/infer where non-train set dropout to 0.
        val = tf.layers.dropout(val, rate=output_drop, noise_shape=noise_shape, training=True, name="val_drop")
    return val

def concat(ts, axis=-1):
    if len(ts) == 1:
        return ts[0]
    return tf.concat(ts, axis=axis, name="v_cat")

#
def repeat_apply(input_t, ff, num, scope_basename, **kwargs):
    x = input_t
    for i in range(num):
        scope_name = scope_basename+str(i)
        with tf.name_scope(scope_name), tf.variable_scope(scope_name):
            x = ff(input_t, **kwargs)
    return x

# =====
# feed forward layers

# Linear Layer
def linear(input_t, output_size, bias=True, **kwargs):
    # parameters
    input_size = get_shape(input_t, -1)
    W = tf.get_variable("W", shape=[input_size, output_size])
    if bias:
        B = tf.get_variable("B", shape=[output_size], initializer=tf.initializers.zeros)
    # calculate
    hidden0 = tf.matmul(input_t, W, name="h_mul")
    if bias:
        hidden0 = tf.add(hidden0, B, name="h_bias")
    return act_and_drop(hidden0, **kwargs)

# Embedding
def embed(input_t, vocab_size, output_size, init_vecs=None, trainable=True, **kwargs):
    # parameters
    if init_vecs is not None:
        E = tf.get_variable("E", shape=[vocab_size, output_size], initializer=tf.constant_initializer(init_vecs), trainable=trainable)
    else:
        if not trainable:
            my_print("Warning: un-trainable random embeddings?")
        E = tf.get_variable("E", shape=[vocab_size, output_size], trainable=trainable)
    # calculate
    val = tf.nn.embedding_lookup(E, input_t, name="h_emb")
    return act_and_drop(val, **kwargs)

# Encoders

# CNN with multiple window sizes
# todo: no masks applied here
def mcnn_enc(input_t, window_sizes, output_size, num_layer, **kwargs):
    # parameters
    if not isinstance(window_sizes, (list, tuple)):
        window_sizes = [window_sizes]
    prev_input = input_t
    #
    one_output_size = output_size // len(window_sizes)
    for cur_layer in range(num_layer):
        input_size = get_shape(prev_input, -1)
        WS = [tf.get_variable(f"W{cur_layer}{i}", shape=[s, input_size, one_output_size]) for i,s in enumerate(window_sizes)]
        BS = [tf.get_variable(f"B{cur_layer}{i}", shape=[one_output_size], initializer=tf.initializers.zeros) for i,s in enumerate(window_sizes)]
        hs = []
        for idx in range(len(window_sizes)):
            conv = tf.nn.conv1d(prev_input, WS[idx], stride=1, padding="SAME", name="h_conv")
            conv_b = tf.nn.bias_add(conv, BS[idx], name="h_conv1")
            conv_h = act_and_drop(conv_b, **kwargs)
            hs.append(conv_h)
        prev_input = tf.concat(hs, -1)
    # no post_process here
    return prev_input

#
def get_rnn_cell(num_units, cell_type, rnn_idrop, rnn_sdrop, residual, input_size=None):
    if cell_type == "lstm":
        cell = tf.nn.rnn_cell.LSTMCell(num_units, name="lstm_cell", dtype=tf.float32)
    elif cell_type == "gru":
        cell = tf.nn.rnn_cell.GRUCell(num_units, name="gru_cell", dtype=tf.float32)
    else:
        raise NotImplementedError()
    #
    if rnn_idrop > 0. or rnn_sdrop > 0.:
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=1.-rnn_idrop, state_keep_prob=1.-rnn_sdrop,
                                             variational_recurrent=True, dtype=tf.float32, input_size=input_size)
    if residual:
        cell = tf.nn.rnn_cell.ResidualWrapper(cell)
    return cell

# num_residual_layer means the last N layers are wrapped by ResidualWrapper
def get_rnn_mcells(num_layer, num_units, cell_type, rnn_idrop, rnn_sdrop, num_residual_layer, input_sizes=None):
    cells = [get_rnn_cell(num_units, cell_type, rnn_idrop, rnn_sdrop, (i>=num_layer-num_residual_layer), input_size=(None if input_sizes is None else input_sizes[i])) for i in range(num_layer)]
    if num_layer == 1:
        return cells[0]
    else:
        return tf.nn.rnn_cell.MultiRNNCell(cells)

# helper
def _rnn_bid_output_helper(outputs, states, bidirection_combine):
    assert len(outputs) == 2
    assert len(states) == 2
    if bidirection_combine:
        enc_out = tf.concat(outputs, axis=-1)
        # todo: may depend too specifically
        if isinstance(states[0], tf.contrib.seq2seq.AttentionWrapperState):
            states = [states[0][0], states[1][0]]
        if isinstance(states[0], tf.nn.rnn_cell.LSTMStateTuple):
            enc_state = [tf.concat([states[0][i], states[1][i]], axis=-1) for i in range(len(states[0]))]
        else:
            enc_state = tf.concat(states, axis=-1)
        outputs, states = enc_out, enc_state
    return outputs, states

# RNN encoders, have various options
# [bs, len, dim] -> ([bs, len, dim], RETURNED-States)
def rnn_enc(input_t, length_t, output_size, cell_type="gru", rnn_idrop=0., rnn_sdrop=0., num_layer=1, num_residual_layer=0, bidirection=False, bidirection_combine=True):
    input_size = get_shape(input_t, -1)
    if bidirection:
        one_output_size = output_size // 2
        input_sizes = [input_size] + [one_output_size] * (num_layer-1)
        fw_cell = get_rnn_mcells(num_layer, one_output_size, cell_type, rnn_idrop, rnn_sdrop, num_residual_layer, input_sizes)
        bw_cell = get_rnn_mcells(num_layer, one_output_size, cell_type, rnn_idrop, rnn_sdrop, num_residual_layer, input_sizes)
        bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_t, sequence_length=length_t, swap_memory=True, dtype=tf.float32)
        enc_out, enc_state = _rnn_bid_output_helper(bi_outputs, bi_states, bidirection_combine)
    else:
        one_output_size = output_size
        fw_cell = get_rnn_mcells(num_layer, one_output_size, cell_type, rnn_idrop, rnn_sdrop, num_residual_layer, input_size)
        enc_out, enc_state = tf.nn.dynamic_rnn(fw_cell, input_t, sequence_length=length_t, swap_memory=True, dtype=tf.float32)
    return enc_out, enc_state

# =====
def get_att_mechanism(att_type, num_units, src_t, src_length_t):
    if att_type == "luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, src_t, memory_sequence_length=src_length_t)
    elif att_type == "scaled_luong":
        attention_mechanism = tf.contrib.seq2seq.LuongAttention(num_units, src_t, memory_sequence_length=src_length_t, scale=True)
    elif att_type == "bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, src_t, memory_sequence_length=src_length_t)
    elif att_type == "normed_bahdanau":
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(num_units, src_t, memory_sequence_length=src_length_t, normalize=True)
    else:
        raise NotImplementedError()
    return attention_mechanism

#
def get_dec_rnn_cell(src_t, src_len_t, att_size, att_type, rnn_size, rnn_type, rnn_idrop, rnn_sdrop, rnn_layer, rnn_residual_layer, input_sizes=None):
    # first the RNN cell
    rnn_cell = get_rnn_mcells(rnn_layer, rnn_size, rnn_type, rnn_idrop, rnn_sdrop, rnn_residual_layer, input_sizes)
    # wrap att?
    if att_type is not None:
        assert src_t is not None and src_len_t is not None
        att_mec = get_att_mechanism(att_type, att_size, src_t, src_len_t)
        cell = tf.contrib.seq2seq.AttentionWrapper(rnn_cell, att_mec, attention_layer_size=att_size, output_attention=True, name="attention")
    else:
        cell = rnn_cell
    return cell

# RNN decoders, possibly accepting conditions with attention
# decoding: teacher-forcing mode decoding, in fact, still works as encoder
# -> return outputs, end-states
def rnn_dec(dec_mode, src_t, src_state, src_len_t, trg_t, trg_len_t, att_size, att_type, rnn_size, rnn_type="gru", rnn_idrop=0., rnn_sdrop=0., num_layer=1, rnn_residual_layer=0, bidirection=False, bidirection_combine=False):
    if dec_mode == "force":
        input_size = get_shape(trg_t, -1) + att_size
        # basically rnn_enc with extra init calculations
        if bidirection:
            input_sizes = [input_size] + [att_size] * (num_layer - 1)
            fw_cell = get_dec_rnn_cell(src_t, src_len_t, att_size, att_type, rnn_size, rnn_type, rnn_idrop, rnn_sdrop,
                                       num_layer, rnn_residual_layer, input_sizes)
            bw_cell = get_dec_rnn_cell(src_t, src_len_t, att_size, att_type, rnn_size, rnn_type, rnn_idrop, rnn_sdrop,
                                       num_layer, rnn_residual_layer, input_sizes)
            # todo: use enc-states instead of 0?
            initial_state_fw = None
            initial_state_bw = None
            bi_outputs, bi_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, trg_t, sequence_length=trg_len_t, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, swap_memory=True, dtype=tf.float32)
            enc_out, enc_state = _rnn_bid_output_helper(bi_outputs, bi_states, bidirection_combine)
        else:
            fw_cell = get_dec_rnn_cell(src_t, src_len_t, att_size, att_type, rnn_size, rnn_type, rnn_idrop, rnn_sdrop,
                                       num_layer, rnn_residual_layer, input_size)
            initial_state_fw = None
            enc_out, enc_state = tf.nn.dynamic_rnn(fw_cell, trg_t, sequence_length=trg_len_t, initial_state=initial_state_fw, swap_memory=True, dtype=tf.float32)
        return enc_out, enc_state
    elif dec_mode == "beam":
        # expand src for beam search
        assert not bidirection
        raise NotImplementedError()
    else:
        raise NotImplementedError()
