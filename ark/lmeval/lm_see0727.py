#

# analysis based on the data obtained from lm_eval

import os
import sys
import json
import numpy as np
import scipy
from scipy.stats import pearsonr, spearmanr
from collections import OrderedDict

# =====
# collecting the info for all types of input
INPUT_TYPES = ["word", "bpe", "character", "morfessor", "morpheme"]
TYPE2ORDER = {k:i for i,k in enumerate(INPUT_TYPES)}
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

# obtain word2freq from raw train data
def get_w2f(cur_lang, cur_type):
    printing("Building train w2f")
    train_file = os.path.join(get_data_dir(cur_lang, cur_type), "train.txt")
    w2f = {}
    with open(train_file) as fd:
        for line in fd:
            for w in line.split():
                w2f[w] = w2f.get(w, 0) + 1
    return w2f

#
def main(cur_lang, type1, type2):
    # load data
    printing("Loading data")
    with open(f"v_{cur_lang}_{type1}.json") as fd:
        vs1 = [json.loads(line) for line in fd]
    with open(f"v_{cur_lang}_{type2}.json") as fd:
        vs2 = [json.loads(line) for line in fd]
    assert len(vs1) == len(vs2), "Unmatched sizes!"
    # get dict
    w2f1 = get_w2f(cur_lang, type1)
    w2f2 = get_w2f(cur_lang, type2)
    # ======
    type1_short, type2_short = ("%-4s"%type1)[:4], ("%-4s"%type2)[:4],
    printing(f"=====\nAnalysis {cur_lang} {type1}({type1_short}) {type2}({type2_short})")
    # analysis
    scores = OrderedDict([
        # negative log likelihood per token
        (f"nll-{type1_short}", [-np.average(x["scores"]) for x in vs1]),
        (f"nll-{type2_short}", [-np.average(x["scores"]) for x in vs2]),
        # sentence length (number of token)
        (f"len-{type1_short}", [len(x["idxes"]) for x in vs1]),
        (f"len-{type2_short}", [len(x["idxes"]) for x in vs2]),
        # character per token
        (f"cpt-{type1_short}", [np.average([len(w) for w in x["tokens"]]) for x in vs1]),
        (f"cpt-{type2_short}", [np.average([len(w) for w in x["tokens"]]) for x in vs2]),
        # oov rate
        (f"oov-{type1_short}", [np.average([int(w not in w2f1) for w in x["tokens"]]) for x in vs1]),
        (f"oov-{type2_short}", [np.average([int(w not in w2f2) for w in x["tokens"]]) for x in vs2]),
    ])
    # correlations
    for k1 in scores:
        if not k1.startswith("nll"): continue
        for k2 in scores:
            if k1==k2: continue
            s1, s2 = scores[k1], scores[k2]
            cor_p, cor_sp = tuple(pearsonr(s1, s2)), tuple(spearmanr(s1, s2))
            printing(f"{k1} vs {k2}: {['%.3f'%x for x in cor_p]} {['%.3f'%x for x in cor_sp]}")

if __name__ == '__main__':
    main(*sys.argv[1:])

# example
# ./python3 lm_see0727.py ess word bpe
