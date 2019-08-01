import argparse
import time
import math
import os
import torch
import torch.nn as nn
# import torch.onnx
import re
import pickle

# import reader
# import RNN
# import word_count

# the original names
import model
from data import Corpus


# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.

def batchify(data, bsz, device):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

def get_batch(my_args, source, i):
    seq_len = min(my_args.bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    # target = source[i + 1:i + 1 + seq_len].view(-1)
    target = source[i + 1:i + 1 + seq_len]  # [seq, batch]
    return data, target


# get argmax correctness
def get_argmax_correctness(my_args, my_model, my_data_source, my_eval_batch_size):
    # Turn on evaluation mode which disables dropout.
    my_model.eval()
    hidden = my_model.init_hidden(my_eval_batch_size)
    rets = [torch.ones(1, my_eval_batch_size).byte()]  # assume the init one is correct and pad one row of True
    with torch.no_grad():
        for i in range(0, my_data_source.size(0) - 1, my_args.bptt):
            data, targets = get_batch(my_args, my_data_source, i)
            output, hidden = my_model(data, hidden)
            _, argmax_ids = output.max(-1)
            rets.append(argmax_ids == targets)
            hidden = repackage_hidden(hidden)
    # return the original flattened sequence of correctness
    return torch.cat(rets, 0).t().contiguous().view(-1).numpy()


def main():
    parser = argparse.ArgumentParser(description='Baseline RNN Language Model')
    parser.add_argument('--data', type=str, default='./data/',
                        help='location of the data corpus')
    parser.add_argument('--test_path', type=str, default=None,
                        help='location of the test corpus to calculate word or character-level perplexity')
    parser.add_argument('--input', type=str, default='word',
                        help='input level (word, grapheme, bpe, syllable, morfessor, char)')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=650,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=650,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=10,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=234,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='model.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx_export', type=str, default='',
                        help='path to export the final model in onnx format')
    # =====
    parser.add_argument("--corr_type", type=str, default="word")  # word, char, bpe
    # =====
    args = parser.parse_args()

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    device = torch.device("cuda" if args.cuda else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    # corpus = reader.Corpus(args.data, args.input)
    corpus = Corpus(args.data)

    eval_batch_size = 10
    # train_data = batchify(corpus.train, args.batch_size, device)
    # val_data = batchify(corpus.valid, eval_batch_size, device)
    test_data = batchify(corpus.test, eval_batch_size, device)

    ###############################################################################
    # Evaluation code
    ###############################################################################

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        if args.cuda:
            model = torch.load(f)
        else:
            model = torch.load(f, map_location='cpu')
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        model.rnn.flatten_parameters()

    # Run on test
    import numpy as np
    correctness = get_argmax_correctness(args, model, test_data, eval_batch_size)
    idx2word = corpus.dictionary.idx2word
    orig_words = [idx2word[z] for z in test_data.t().contiguous().view(-1).numpy()]
    assert len(orig_words) == len(correctness)
    my_eval(correctness, orig_words, args.corr_type)

# =====
def my_eval(correctness, orig_words, corr_type):
    if corr_type == "word":
        end_of_words = [1] * len(correctness)  # every word is an end of word
    elif corr_type == "char":
        # using "_" as breaker
        end_of_words = [int(x=="_" or x=="<eos>" or (idx<len(orig_words)-1 and orig_words[idx+1]=="<eos>"))
                        for idx, x in enumerate(orig_words)]
    else:
        # using "@@" as connectors
        end_of_words = [int(not x.endswith("@@")) for x in orig_words]
    end_of_words[-1] = 1  # force the ending
    # evaluate
    orig_count, word_count = len(end_of_words), sum(end_of_words)
    orig_corr_count, word_corr_count = 0, 0
    cont_corr = True
    for cur_corr, cur_eow in zip(correctness, end_of_words):
        cur_corr = bool(cur_corr)
        if cur_corr:
            orig_corr_count += 1
        cont_corr = (cont_corr and cur_corr)
        if cur_eow:
            if cont_corr:
                word_corr_count += 1
            cont_corr = True
    orig_acc, word_acc = orig_corr_count/orig_count, word_corr_count/word_count
    print("Orig: %s/%s=%s, Word: %s/%s=%s" % (orig_corr_count, orig_count, orig_acc, word_corr_count, word_count, word_acc))
# =====

if __name__ == '__main__':
    main()

# README
# args
"""
# --corr_type corresponds to what type of correction we are evaluating (word, char, bpe)
# --data is the dir which contains the "test.txt" test file
# --save is the pre-trained model path
"""
# example running cmd
"""
python3 e.py --corr_type word --data /home/jsalt2019_kandykan/word_lm/data/grn_bpe_10k/ --save /home/jsalt2019_kandykan/word_lm/baseline_models/grn_bpe_10k.pt
"""
# real running script
"""
for f in /home/jsalt2019_kandykan/word_lm/baseline_models/*; do
orig_model_name=`basename $f .pt`
model_name=${orig_model_name}
# which model?
if echo ${model_name} | grep word >/dev/null; then
model_type="word"
model_name=`basename $f _word.pt`  # further strip
elif echo ${model_name} | grep char >/dev/null; then
model_type="char"
else
model_type="bpe"
fi
# run
echo "RUN ${model_type} model: ${model_name}"
python3 e.py --corr_type ${model_type} --data /home/jsalt2019_kandykan/word_lm/data/${model_name}/ --save /home/jsalt2019_kandykan/word_lm/baseline_models/${orig_model_name}.pt
done |& tee log
#
cat log | grep -E "RUN|Orig"
"""
