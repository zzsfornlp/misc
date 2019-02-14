#

import tensorflow as tf
from tensorflow.python import debug as tf_debug
from .nntf import TFMode
from .utils import my_print

#
class Model:
    def __init__(self, args):
        self.args = args
        # build graphs/models for train/eval/infer
        if args.tf_device >= 0:
            device_descr = "/gpu:%d" % args.tf_device
        else:
            device_descr = "/cpu:0"
        with tf.device(device_descr):
            self.packs = {}
            for mode in [TFMode.TRAIN, TFMode.EVAL, TFMode.INFER]:
                self.packs[mode] = self._get_pack(mode)

    # from nmt-tutorial
    def _get_config_proto(self, log_device_placement=False, allow_soft_placement=True,
                         num_intra_threads=0, num_inter_threads=0):
        # GPU options:
        # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
        config_proto = tf.ConfigProto(log_device_placement=log_device_placement,
                                      allow_soft_placement=allow_soft_placement)
        config_proto.gpu_options.allow_growth = True
        # CPU threads options
        if num_intra_threads:
            config_proto.intra_op_parallelism_threads = num_intra_threads
        if num_inter_threads:
            config_proto.inter_op_parallelism_threads = num_inter_threads
        return config_proto

    def _get_session(self, config, graph, debug):
        sess = tf.Session(config=config, graph=graph)
        if debug:
            my_print("Using debugging mode!")
            sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        return sess

    def _get_pack(self, mode):
        args = self.args
        #
        graph = tf.Graph()
        with graph.as_default():
            inputs, outputs = self._build_graph(mode)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.tf_num_keep_ckpts)
        config_proto = self._get_config_proto(
            log_device_placement=args.tf_log_device_placement,
            num_intra_threads=args.tf_num_intra_threads,
            num_inter_threads=args.tf_num_inter_threads)
        sess = self._get_session(config=config_proto, graph=graph, debug=args.tf_debug)
        return {"mode": mode, "graph": graph, "sess": sess, "inputs": inputs, "outputs": outputs, "saver": saver}

    # =====
    # todo(warn): to implement
    # todo(warn): make sure every mode has the same parameters
    def _build_graph(self, mode):
        apply_dropout = (mode==TFMode.TRAIN)
        def _d(orig_dropout):
            return orig_dropout if apply_dropout else 0.
        # inputs
        # models
        # outputs
        raise NotImplementedError()

    def _build_inputs(self, mode, insts):
        raise NotImplementedError()
    # =====

    #
    def _run(self, mode, insts):
        feed_dict = self._build_inputs(mode, insts)
        pack = self.packs[mode]
        sess = pack["sess"]
        return sess.run(fetches=pack["outputs"], feed_dict=feed_dict)

    def train(self, insts): return self._run(TFMode.TRAIN, insts)
    def eval(self, insts): return self._run(TFMode.EVAL, insts)
    def infer(self, insts): return self._run(TFMode.INFER, insts)

    #
    def load_or_init(self, fname, mode=TFMode.TRAIN):
        try:
            self.load(fname, mode)
        except:
            my_print("Cannot load previous models, init from scratch instead.")
            self.init(mode)

    def init(self, mode=TFMode.TRAIN):
        pack = self.packs[mode]
        with pack["graph"].as_default():
            sess = pack["sess"]
            sess.run(tf.global_variables_initializer())
            sess.run(tf.tables_initializer())
            my_print(f"Model({mode}) init")

    def load(self, fname, mode=TFMode.TRAIN):
        pack = self.packs[mode]
        with pack["graph"].as_default():
            pack["saver"].restore(pack["sess"], fname)
            my_print(f"Model({mode}) restored from {fname}")

    def save(self, fname, mode=TFMode.TRAIN):
        pack = self.packs[mode]
        with pack["graph"].as_default():
            save_path = pack["saver"].save(pack["sess"], fname)
            my_print(f"Model({mode}) saved in path: {save_path}")

if __name__ == '__main__':
    pass
