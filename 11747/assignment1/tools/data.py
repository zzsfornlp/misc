#

import json
import numpy as np
from .utils import my_open, my_print, shuffle_files

# =====
# Part 1: Vocabulary

# word level special token
UNK = "<unk>"
SOS = "<s>"
EOS = "</s>"
UNK_ID = 0
SOS_ID = 1
EOS_ID = 2

#
def file_token_stream(fname, sep=" "):
    with my_open(fname) as fd:
        for line in fd:
            line = line.rstrip()
            for tok in line.split(sep):
                yield tok

#
class Vocab:
    def __init__(self, word_types, word_freqs, name="anon"):
        # first adding specials
        # self.wordtypes = word_types
        self.name = name
        self.special_words = [UNK, SOS, EOS]
        self.id2word = self.special_words + word_types
        self.id2freq = [100] * len(self.special_words) + word_freqs
        self.word2id = {w: i for i,w in enumerate(self.id2word)}
        assert len(self.word2id) == len(self.id2word), "Repeated words types!!"
        assert len(self.word2id) == len(self.id2freq), "Unmatched freqs!!"
        my_print(f"Vocab: Finally, build Vocab of size {len(self)}.")

    def i2w(self, idx):
        return self.id2word[idx]

    def i2f(self, idx):
        return self.id2freq[idx]

    def w2i(self, w, default=UNK_ID):
        return self.word2id.get(w, default)

    # filter embeds from gensim model
    def filter_embed(self, emb_size, g_model, init_nohit=0.2):
        mm = g_model.wv
        hit = 0
        hit_lower = 0
        vecs = []
        for w in self.id2word:
            if w in mm:
                vec = mm[w]
                hit += 1
            elif str.lower(w) in mm:
                vec = mm[str.lower(w)]
                hit_lower += 1
            else:
                # random init, todo: random or rand-normal?
                # vec = (np.random.random_sample(emb_size).astype(np.float32)-0.5) * (2*init_nohit)
                vec = np.random.randn(emb_size).astype(np.float32) * init_nohit
            vecs.append(vec)
        my_print(f"All/hit/hit-lower = {len(vecs)}/{hit}/{hit_lower}")
        return np.asarray(vecs)

    def __getitem__(self, item):
        return self.word2id[item]

    def __len__(self):
        return len(self.id2word)

    def __repr__(self):
        return f"Vocab({self.name}, len={len(self)})"

    @staticmethod
    def build_from_streams(streams, max_word, min_freq):
        # todo(warn): filtering-criteria as `len(vocab)<=max_word+SPECIAL_NUM, freq(w)>=min_freq'
        # get raw counts
        # todo(warn): only count for the first-stream(train)
        cc = {}
        for s in streams[0]:
            count = cc.get(s, 0)
            cc[s] = count + 1
        # add extra streams: only adding
        for extra_stream in streams[1:]:
            for s in extra_stream:
                if s not in cc:
                    cc[s] = 1
        # filter words
        ranked_words = sorted(cc.keys(), key=lambda w: -cc[w])
        cutting_length = min(max_word, len(ranked_words))
        while cutting_length>0 and cc[ranked_words[cutting_length-1]]<min_freq:
            cutting_length -= 1
        filtered_words = ranked_words[:cutting_length]
        filtered_freqs = [cc[w] for w in filtered_words]
        my_print(f"Vocab: Cutting with ``#Word<={max_word} && #Freq>={min_freq}'':"
                 f" from {len(ranked_words)} to {len(filtered_words)}")
        return Vocab(filtered_words, filtered_freqs)

    @staticmethod
    def build_from_files(fnames, sep=" ", **kwargs):
        # todo(warn): build from TextLineFile, tokenized by sep
        return Vocab.build_from_streams([file_token_stream(f, sep) for f in fnames], **kwargs)

    @staticmethod
    def read_from_file(fname):
        v = Vocab([], [])
        v.load(fname)
        return v

    def save(self, fname):
        with my_open(fname, "w") as fd:
            json.dump([self.word2id, self.id2word, self.id2freq], fd)
            my_print(f"Vocab: save {self} to {fname}")

    def load(self, fname):
        with my_open(fname) as fd:
            self.word2id, self.id2word, self.id2freq = json.load(fd)
            my_print(f"Vocab: load {self} from {fname}")


# =====
# Part 2: data iter

class ValueReader:
    def __init__(self, t):
        self.t = t

    def __call__(self, fd, *args):
        for line in fd:
            yield [self.t(z) for z in line.strip().split()]

class TextReader:
    def __init__(self, v, unk_repl=0., unk_repl_thresh=0, sep=" "):
        self.v = v
        self.unk_repl = unk_repl
        self.unk_repl_thresh = unk_repl_thresh
        self.sep = sep
        #
        self.use_unk_repl = (unk_repl>0. and unk_repl_thresh>0)
        self.repl_rates = np.asarray([(0. if v.i2f(idx)>unk_repl_thresh else unk_repl/v.i2f(idx)) for idx in range(len(v))])

    def __call__(self, fd, *args):
        for line in fd:
            # todo(warn): add sos/eos here!
            ret = [SOS_ID] + [self.v.w2i(w) for w in line.strip().split(self.sep)] + [EOS_ID]
            if self.use_unk_repl:
                cur_len = len(ret)
                repl_rates = self.repl_rates[ret]
                whether_repl = (np.random.random_sample(cur_len) < repl_rates)
                ret = [(UNK_ID if rr else z) for z, rr in zip(ret, whether_repl)]
            yield ret

#
def next_or_none(s):
    try:
        return next(s)
    except StopIteration:
        return None

# -- Read data and iterate parallel data to batches (return numpy arrs)
# -- Infinite stream, cur by outside force!!
# files: parallel datas, can be plain-tokenized-text or other types, BUT MUST be one line per instance!
# data_types: text, arr, scalar
# vocabs: vocabs for text, None for others
def iter_data_batched(files, data_types, vocabs, batch_size=128, maxlen=80, shuffle_each_epoch=False, shuffle_bucket=True, sort_by_length=True, maxibatch_size=20, max_epoch=-1, eoe_none=False, unk_repl=0., unk_repl_thresh=0):
    #
    num_data = len(files)
    assert len(data_types) == num_data
    assert len(vocabs) == num_data
    orig_files = files
    #
    if shuffle_each_epoch:
        files = shuffle_files(orig_files)
    # loop
    cur_epoch = 0
    is_text = [dt=="text" for dt in data_types]
    read_fs = [{"text": lambda fd, vv: TextReader(vv, unk_repl=unk_repl, unk_repl_thresh=unk_repl_thresh)(fd),
                "float": ValueReader(float),
                "int": ValueReader(int)}[dt] for dt in data_types]
    fds = [my_open(f) for f in files]
    streamers = [f(fd, vv) for f, fd, vv in zip(read_fs, fds, vocabs)]
    #
    k = batch_size * maxibatch_size
    keep_goging = True
    while keep_goging:
        pending_eos = False
        # collect cached data
        cache = []
        while len(cache) < k:
            para_data = [next_or_none(st) for st in streamers]
            end_of_data = [z is None for z in para_data]
            if all(end_of_data):
                # reset
                cur_epoch += 1
                if max_epoch>=0 and cur_epoch>=max_epoch:
                    my_print(f"End of iteration before the start of cur-epoch = {cur_epoch}")
                    keep_goging = False
                    break
                if eoe_none:
                    pending_eos = True      # end-of-epoch mark
                my_print(f"Iter: reset on new epoch #{cur_epoch}!")
                if shuffle_each_epoch:
                    files = shuffle_files(orig_files)
                fds = [my_open(f) for f in files]
                streamers = [f(fd, vv) for f, fd, vv in zip(read_fs, fds, vocabs)]
                break
            else:
                assert not any(end_of_data), "Unmatched data!"
                # filter by length
                if not any(t and len(d)>maxlen for t,d in zip(is_text, para_data)):
                    cache.append(para_data)
        # make them into batches
        # todo(warn): sorting by the first-len (should usually be src-text)
        if sort_by_length:
            cache.sort(key=lambda x: len(x[0]))
        # prepare all the buckets
        buckets = []
        cur_idx = 0
        while cur_idx < len(cache):
            cur_slice = cache[cur_idx:cur_idx+batch_size]
            cur_data = []
            for idx in range(num_data):
                cur_pieces = [z[idx] for z in cur_slice]
                if is_text[idx]:
                    # padding text with EOS
                    cur_lengths = list(map(len, cur_pieces))
                    cur_max_len = max(cur_lengths)
                    for one in cur_pieces:
                        if len(one) < cur_max_len:
                            one.extend([EOS_ID]*(cur_max_len-len(one)))
                    # (idxes: [bs, slen], lengths: [bs])
                    # todo(warn): augmenting with lengths, or using masks?
                    cur_data.append([np.asarray(cur_pieces), np.asarray(cur_lengths)])
                else:
                    # asserting matched length
                    assert all(len(z)==len(cur_pieces[0]) for z in cur_pieces)
                    # (values: [bs, vlen])
                    cur_data.append(np.asarray(cur_pieces))
            buckets.append(cur_data)
            cur_idx += batch_size
        # yield them
        if shuffle_bucket:
            np.random.shuffle(buckets)
        for one in buckets:
            yield one
        if pending_eos:
            yield None
