#

#
import sys
import platform
sys.path.append("..")
from typing import Sequence
from collections import defaultdict
import numpy as np
import gensim

#
from tools.conf import Conf
from tools.data import Vocab, iter_data_batched
from tools.model import Model
from tools import nntf
from tools.nntf import TFMode
import tensorflow as tf
from tools.run import Runner, StatusRecorder
from tools.utils import my_print, my_open, Timer

# =====
HW1_CLASSES = ['UNK', 'Agriculture, food and drink', 'Art and architecture', 'Engineering and technology', 'Geography and places', 'History', 'Language and literature', 'Mathematics', 'Media and drama', 'Miscellaneous', 'Music', 'Natural sciences', 'Philosophy and religion', 'Social sciences and society', 'Sports and recreation', 'Video games', 'Warfare']
HW1_VOCAB = {n:i for i,n in enumerate(HW1_CLASSES)}
# =====

#
class Hw1Conf(Conf):
    def __init__(self, args):
        super().__init__(None)
        # model
        self.dim_word = 300
        self.enc_size = 300
        self.enc_type = "cnn"       # 'cnn', 'lstm', 'gru'
        self.enc_layer = 1
        self.enc_cnn_windows = [3, 4, 5]
        self.enc_rnn_bidirection = True         # only for rnn type: bidirectional rnn type
        self.dec_type = "none"       # 'none', 'lstm', 'gru'
        self.dec_layer = 1
        self.dec_size = 300
        self.dec_rnn_att_type = 'luong'
        self.dec_rnn_att_size = 300
        self.dec_rnn_bidirection = True
        self.scorer_layer = 0           # extra hidden layer
        self.scorer_size = 300          # size of extra hidden layer
        self.output_size = len(HW1_CLASSES)            # how many classes at output?
        # others
        self.drop_embed = 0.5
        self.drop_hidden = 0.5
        self.idrop_rnn = 0.33
        self.sdrop_rnn = 0.33
        self.unk_repl = 0.5             # unk replacement
        self.unk_repl_thresh = 2        # <= this one
        self.pretrain = ""
        # =====
        # override default
        self.trainer_type = "adam"
        self.lrate = 0.0002
        # =====
        #
        if args is not None:
            self.init_from(args)
        #
        # to be filled!!
        self.vocab_size = None
        self.embed_init = None

    # dynamically get other non-settable confs
    def setup(self, vocab):
        self.vocab_size = len(vocab)
        self.embed_init = None
        if self.pretrain:
            my_print("Reading w2v-pretrained file " + self.pretrain)
            go_binary = "bin" in self.pretrain
            model = gensim.models.KeyedVectors.load_word2vec_format(self.pretrain, binary=go_binary)
            self.embed_init = vocab.filter_embed(self.dim_word, model)
        pass

#
class Hw1Model(Model):
    def __init__(self, conf: Hw1Conf):
        super().__init__(conf)

    def _build_graph(self, mode):
        conf = self.args
        #
        is_training = (mode==TFMode.TRAIN)
        is_evaling = (mode==TFMode.EVAL)
        is_inferring = (mode==TFMode.INFER)
        apply_dropout = is_training
        def _d(orig_dropout):
            return orig_dropout if apply_dropout else 0.
        # inputs
        input_text = tf.placeholder(dtype=tf.int32, shape=[None, None], name='input_text')
        input_text_length = tf.placeholder(dtype=tf.int32, shape=[None, ], name='input_text_length')
        inputs = {"input_text": input_text, "input_text_length": input_text_length}
        if not is_inferring:
            input_label = tf.placeholder(dtype=tf.int32, shape=[None, ], name='input_label')
            inputs["input_label"] = input_label
            input_label_1hot = tf.one_hot(input_label, self.args.output_size)
        if is_training:
            input_lrate = tf.placeholder(dtype=tf.float32, shape=(), name='input_lrate')
            inputs["input_lrate"] = input_lrate
        # models
        # embeddings
        with tf.variable_scope("embed"), tf.name_scope("embed"):
            embed_t = nntf.embed(input_text, conf.vocab_size, conf.dim_word, init_vecs=conf.embed_init, output_drop=_d(conf.drop_embed))
        # encoder
        with tf.variable_scope("encode"), tf.name_scope("encode"):
            if conf.enc_type == "cnn":
                enc_outputs = nntf.mcnn_enc(embed_t, conf.enc_cnn_windows, conf.enc_size, conf.enc_layer, output_act='relu', output_drop=_d(conf.drop_hidden))
                # enc_outputs = nntf.mcnn_enc(embed_t, conf.enc_cnn_windows, conf.enc_size, conf.enc_layer, output_act='relu')
                enc_states = tf.reduce_max(enc_outputs, axis=-2)       # max pooling [bs, dim]
            else:
                # using bidirection rnn
                enc_outputs, enc_states = nntf.rnn_enc(embed_t, input_text_length, conf.enc_size, cell_type=conf.enc_type, rnn_idrop=_d(conf.idrop_rnn), rnn_sdrop=_d(conf.sdrop_rnn), num_layer=conf.enc_layer, bidirection=conf.enc_rnn_bidirection, bidirection_combine=True)
                if isinstance(enc_states, Sequence):
                    enc_states = enc_states[0]      # only using h if lstm
        # decoder
        with tf.variable_scope("decoder"), tf.name_scope("decoder"):
            if conf.dec_type == "none":
                dec_outputs = enc_outputs
            else:
                # another attentional rnn
                dec_outputs, _ = nntf.rnn_dec("force", enc_outputs, enc_states, input_text_length, embed_t, input_text_length, att_size=conf.dec_rnn_att_size, att_type=conf.dec_rnn_att_type, rnn_size=conf.dec_size, rnn_type=conf.dec_type, rnn_idrop=_d(conf.idrop_rnn), rnn_sdrop=_d(conf.sdrop_rnn), num_layer=conf.dec_layer, bidirection=conf.dec_rnn_bidirection, bidirection_combine=True)
        # final scorer
        with tf.variable_scope("scorer"), tf.name_scope("scorer"):
            # max pooling & dropout
            sent_repr = tf.reduce_max(dec_outputs, axis=-2)
            sent_repr = nntf.act_and_drop(sent_repr, output_drop=_d(conf.drop_hidden))
            #
            for i in range(conf.scorer_layer):
                with tf.variable_scope(f"scorer_h{i}"), tf.name_scope(f"scorer_h{i}"):
                    sent_repr = nntf.linear(sent_repr, conf.scorer_size, output_act='relu', output_drop=_d(conf.drop_hidden))
            final_scores = nntf.linear(sent_repr, self.args.output_size)
            # final_probs = tf.sigmoid(final_scores, name="prob")
            # final_predicts = tf.to_float(final_probs>0.5, name="predict")
            final_probs = tf.nn.softmax(final_scores, axis=-1, name="prob")
            final_predicts = tf.to_int32(tf.argmax(input=final_scores, axis=-1))
        # outputs
        if is_inferring:
            outputs = {"probs": final_probs, "predicts": final_predicts}
        else:
            # loss
            # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=input_label, logits=final_scores)
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_label_1hot, logits=final_scores)
            loss = tf.reduce_mean(losses, axis=0)
            outputs = {"probs": final_probs, "predicts": final_predicts, "loss": loss, "correct": tf.equal(final_predicts, input_label)}
            # training
            if is_training:
                opter = {
                    "sgd": tf.train.GradientDescentOptimizer,
                    "momentum": lambda lr: tf.train.MomentumOptimizer(lr, conf.momentum),
                    "adagrad": tf.train.AdagradOptimizer,
                    "adam": tf.train.AdamOptimizer,
                }[conf.trainer_type](input_lrate)
                gradients, variables = zip(*opter.compute_gradients(loss))
                gradients, _ = tf.clip_by_global_norm(gradients, conf.clip_c)
                op = opter.apply_gradients(zip(gradients, variables))
                outputs["op"] = op
        return inputs, outputs

    def _build_inputs(self, mode, insts):
        is_training = (mode == TFMode.TRAIN)
        is_evaling = (mode == TFMode.EVAL)
        is_inferring = (mode == TFMode.INFER)
        # prepare the inputs
        feed_dict = {"input_text": insts[0][0], "input_text_length": insts[0][1]}
        if not is_inferring:
            feed_dict["input_label"] = np.reshape(insts[1], [-1])
        if is_training:
            feed_dict["input_lrate"] = insts[2]     # todo(warn): put together as inputs
        # translate to tf
        input_dict = self.packs[mode]["inputs"]
        return {input_dict[n]:v for n,v in feed_dict.items()}

#
class Hw1Runner(Runner):
    def _validate_them(self, dev_iter, metrics):
        # get acc and loss
        loss, loss_num = 0, 0
        count_all, count_corr = 0, 0
        for insts in dev_iter:
            if insts is None:
                break
            res = self._mm.eval(insts)
            count_all += len(res["correct"])
            count_corr += sum(res["correct"])
            loss += res["loss"]
            loss_num += 1
        rr = {"acc": count_corr/count_all, "loss": loss/loss_num}
        return [rr[z] for z in metrics]

    def _test_them(self, test_iter, output_name):
        results = []
        for insts in test_iter:
            if insts is None:
                break
            res = self._mm.infer(insts)
            results.extend([int(z) for z in res["predicts"]])
        with my_open(output_name, 'w') as fd:
            fd.write("\n".join([str(z) for z in results]))

    def _get_recorder(self, name):
        return Hw1StatusRecorder(name)

    def _fb_once(self, insts, cur_lrate):
        res = self._mm.train(insts+[cur_lrate])
        # todo(warn); roughly
        ret = {"main": res["loss"], "acc": sum(res["correct"])/len(res["correct"])}
        return ret

#
class Hw1StatusRecorder(StatusRecorder):
    def __init__(self, name, mm=None):
        self.name = name
        self.loss = defaultdict(float)
        self.sents = 1e-6
        self.words = 1e-6
        self.updates = 0
        self.timer = Timer("")
        self._mm = mm

    def record(self, insts, loss, update):
        for k in loss:
            self.loss[k] += loss[k]
        self.sents += len(insts[0][0])
        self.words += sum(insts[0][1])
        self.updates += update

    def reset(self):
        self.loss = self.loss = defaultdict(float)
        self.sents = 1e-5
        self.words = 1e-5
        self.updates = 0
        self.timer = Timer("")

    # const, only reporting, could be called many times
    def state(self):
        one_time = self.timer.get_time()
        loss_per_update = "|".join(["%s:%.4f" % (k, self.loss[k] / self.updates) for k in sorted(self.loss.keys())])
        loss_per_sentence = "|".join(["%s:%.4f" % (k, self.loss[k] / self.sents) for k in sorted(self.loss.keys())])
        loss_per_word = "|".join(["%s:%.4f" % (k, self.loss[k] / self.words) for k in sorted(self.loss.keys())])
        update_per_second = float(self.updates) / one_time
        sent_per_second = float(self.sents) / one_time
        word_per_second = float(self.words) / one_time
        return f"Recoder {self.name}, time={one_time:.2f}/{update_per_second:.2f}/{sent_per_second:.2f}/{word_per_second:.2f}, num={self.updates:.1f}/{self.sents:.1f}/{self.words:.1f}, loss={loss_per_update}/{loss_per_sentence}/{loss_per_word}"

    def report(self, s=""):
        my_print(s + self.state())

# =====
# running commands
def hw1_train(args):
    # 0. conf
    conf = Hw1Conf(args)
    # 1. vocab
    if conf.no_rebuild_vocab:
        vocab = Vocab.read_from_file(conf.vocab)
    else:
        vocab = Vocab.build_from_files([z+".text" for z in [conf.train, conf.dev, conf.test] if z], sep=" ", max_word=conf.vocab_rthres, min_freq=conf.vocab_fthres)
        vocab.save(conf.vocab)
    # 2. model
    conf.setup(vocab)
    model = Hw1Model(conf)
    # 3. init model
    if conf.reload_model:
        model.load_or_init(conf.model)
    else:
        model.init()
    # 4. data
    train_iter = iter_data_batched([conf.train+".text", conf.train+".label"], ["text", "int"], [vocab, None],
                                   batch_size=conf.batch_size, maxlen=conf.max_len, shuffle_each_epoch=conf.shuffle_each_epoch,
                                   shuffle_bucket=True, sort_by_length=True, max_epoch=conf.max_epochs, eoe_none=True,
                                   unk_repl=conf.unk_repl, unk_repl_thresh=conf.unk_repl_thresh)
    dev_iter = iter_data_batched([conf.dev+".text", conf.dev+".label"], ["text", "int"], [vocab, None],
                                 batch_size=conf.batch_size, maxlen=9999999, shuffle_each_epoch=False,
                                 shuffle_bucket=True, sort_by_length=True, max_epoch=-1, eoe_none=True)
    # 5. go!
    trainer = Hw1Runner(conf, model)
    trainer.train(train_iter, dev_iter)

#
def hw1_test(args):
    # 0. conf
    conf = Hw1Conf(args)
    # 1. load vocab
    vocab = Vocab.read_from_file(conf.vocab)
    # 2. model
    conf.setup(vocab)
    model = Hw1Model(conf)
    # 3. load model (plus best suffix!!)
    model.load(conf.model+Runner.BEST_SUFFIX)
    # 4. data (keep the order)
    test_iter = iter_data_batched([conf.test+".text"], ["text"], [vocab],
                                 batch_size=conf.batch_size, maxlen=9999999, shuffle_each_epoch=False,
                                 shuffle_bucket=False, sort_by_length=False, max_epoch=1, eoe_none=False)
    # 5. go
    tester = Hw1Runner(conf, model)
    tester.test(test_iter)
    # 6. evaluate
    gold_file = conf.gold
    pred_file = conf.output
    if gold_file:
        with my_open(pred_file) as pfd, my_open(gold_file) as gfd:
            preds = [int(z) for z in pfd]
            golds = [int(z) for z in gfd]
            num_all = len(golds)
            assert len(preds) == num_all
            num_corr = sum(1 if a==b else 0 for a,b in zip(preds,golds))
            my_print(f"Evaluate: {num_corr}/{num_all}={num_corr/num_all}")
#
if __name__ == '__main__':
    nntf.init_seed()
    #
    if platform.system() == "Windows":
        DEBUG = 1
    else:
        DEBUG = 0
    if DEBUG:
        # debug
        hw1_train("train:data/topicclass_valid.txt dev:data/topicclass_valid.txt".split())
        hw1_test("test:data/topicclass_test.txt".split())
    else:
        if sys.argv[1] == "train":
            hw1_train(sys.argv[2:])
        elif sys.argv[1] == "test":
            hw1_test(sys.argv[2:])
        else:
            raise NotImplementedError()

# python3 hw1.py train train:data/train.txt dev:data/test.txt pretrain:./data/emb.txt
# python3 ../hw1/hw1.py train train:data/topicclass_train.txt dev:data/topicclass_valid.txt pretrain:./data/emb.txt
