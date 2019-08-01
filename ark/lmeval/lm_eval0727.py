#

# get the log scores
# based on models trained with https://github.com/salesforce/awd-lstm-lm

import os
import sys
import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import hashlib
import glob
import json

# =====
# from awd-lstm-lm
import data
import model
# =====

# =====
# collecting the info for all types of input
INPUT_TYPES = ["word", "bpe", "character", "morfessor", "morpheme"]
LANGS = "ess esu iku grn".split()

def get_dir_name(lang, type):
    if type == "word":
        return f"Lang.{lang}"
    else:
        return f"Lang.{lang}+Tokenize.{type}"

def get_data_dir(lang, type):
    return f"/home/hpark129/projects/baseline_RNN/results/tokenize/{get_dir_name(lang, type)}"

def get_model_path(lang, type):
    return f"/home/hpark129/projects/baseline_RNN/results/train/{get_dir_name(lang, type)}/model.pt"

def printing(s):
    print(s, file=sys.stderr)

# =====
# evaluation
def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def evaluate(data_dir, model_path, batch_size, chunk_size, use_cuda):
    # =====
    printing(f"Evaluating {data_dir} & {model_path}")
    # read corpus, especially for vocab
    model_dir = os.path.dirname(model_path)
    # fn = 'corpus.{}.data'.format(hashlib.md5(data_dir.encode()).hexdigest())
    fn = glob.glob(f"{model_dir}/corpus.*.data")
    assert len(fn)==1
    fn = fn[0]
    printing(f"Loading dataset from {fn}")
    corpus = torch.load(fn)
    word2idx = corpus.dictionary.word2idx
    # load model
    with open(model_path, 'rb') as f:
        printing(f"Loading model from {f}")
        model, criterion, optimizer = torch.load(f)
    if use_cuda:
        model = model.cuda()
    else:
        model = model.cpu()
    # Turn on evaluation mode which disables dropout.
    model.eval()
    # =====
    # read test and eval
    test_file = os.path.join(data_dir, "test.txt")
    test_data = []
    with open(test_file) as fd:
        for sid, line in enumerate(fd):
            tokens = line.split() + ['<eos>']
            # there will be no oov since test is included in vocab when building?
            idxes = [word2idx[w] for w in tokens]
            one = {"sid": sid, "tokens": tokens, "idxes": idxes}
            test_data.append(one)
    # start to decode
    printing(f"Decoding with {len(test_data)} lines of data")
    # sort by length
    test_data.sort(key=lambda x: len(x["idxes"]))
    # batched evaluation
    bidx = 0
    while bidx < len(test_data):
        next_bidx = min(len(test_data), bidx + batch_size)
        cur_data = test_data[bidx:next_bidx]
        bsize = len(cur_data)
        max_length = max([len(x["idxes"]) for x in cur_data])
        # batch, 0 as padding
        cur_data_t = torch.full([bsize, max_length], 0, dtype=torch.long)
        for b in range(bsize):
            one_input = cur_data[b]["idxes"]
            cur_data_t[b][:len(one_input)] = torch.as_tensor(one_input)
        cur_data_t = cur_data_t.t().contiguous()  # [max-length, bsize]
        if use_cuda:
            cur_data_t = cur_data_t.cuda()
        # loop
        logprobs = [torch.full([1, bsize], 0., dtype=torch.float32)]  # start from the first token, but does not predict it
        hidden = model.init_hidden(bsize)
        for start_idx in range(0, cur_data_t.size(0)-1, chunk_size):
            end_idx = min(start_idx+chunk_size, cur_data_t.size(0)-1)
            cur_input_t = cur_data_t[start_idx:end_idx]
            cur_target_t = cur_data_t[start_idx+1:end_idx+1]
            output, hidden = model(cur_input_t, hidden)
            output = model.decoder(output)
            hidden = repackage_hidden(hidden)
            # get log probs
            output_logprobs = torch.nn.functional.log_softmax(output.view(end_idx-start_idx, bsize, -1), -1)
            cur_logprobs = output_logprobs.gather(-1, cur_target_t.unsqueeze(-1)).squeeze(-1)  # [len, bsize]
            logprobs.append(cur_logprobs)
        bidx = next_bidx
        # get the scores back
        final_logprobs = torch.cat(logprobs, 0).t().contiguous()  # [bsize, max-length]
        if use_cuda:
            final_logprobs = final_logprobs.cpu()
        for v, d in zip(final_logprobs, cur_data):
            d["scores"] = v[:len(d["idxes"])].tolist()
    # return
    test_data.sort(key=lambda x: x["sid"])
    return test_data

#
def main(cur_lang, cur_type):
    batch_size = 32
    chunk_size = 80
    use_cuda = False
    with torch.no_grad():
        vs = evaluate(get_data_dir(cur_lang, cur_type), get_model_path(cur_lang, cur_type), batch_size, chunk_size, use_cuda)
    with open(f"v_{cur_lang}_{cur_type}.json", 'w') as fd:
        for one in vs:
            fd.write(json.dumps(one)+"\n")

#
if __name__ == '__main__':
    main(*sys.argv[1:])

# ./python3 lm_eval0727.py ess word
"""
for lang in ess esu iku grn; do
for type in word bpe character morfessor morpheme; do
OMP_NUM_THREADS=2 ./python3 lm_eval0727.py ${lang} ${type}
done
done |& tee log
"""
