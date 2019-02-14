#

# filter embeddings

import gensim
import sys
sys.path.append("..")

from tools.data import Vocab
from tools.utils import my_print, my_open

def main(input, output, *files):
    # build vocab
    vocab = Vocab.build_from_files([z + ".text" for z in files], sep=" ", max_word=1000000, min_freq=0)
    #
    my_print("Reading w2v-pretrained file " + input)
    go_binary = "bin" in input
    model = gensim.models.KeyedVectors.load_word2vec_format(input, binary=go_binary)
    # filter
    filtered_words = set()
    for w in vocab.id2word:
        if w in model.wv:
            filtered_words.add(w)
        elif str.lower(w) in model.wv:
            filtered_words.add(str.lower(w))
    # write
    my_print(f"From origin MANY to filtered {len(filtered_words)}.")
    with my_open(output, 'w') as fd:
        # todo(warn): get dimension by the word 'the'
        fd.write(f"{len(filtered_words)} {len(model.wv['the'])}\n")
        for w in filtered_words:
            fd.write(" ".join([w] + ["%.6f"%z for z in model.wv[w]]) + "\n")

if __name__ == '__main__':
    main(*sys.argv[1:])

# python3 prep_embed.py data/GoogleNews-vectors-negative300.bin.gz data/emb.txt data/{train,dev,test}.txt |& tee data/emb.log
# python3 prep_embed.py data/GoogleNews-vectors-negative300.bin.gz data/emb.txt data/topicclass_{train,valid,test}.txt |& tee data/emb.log
