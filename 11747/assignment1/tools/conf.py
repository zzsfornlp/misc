#

# global configurations

import json
from .utils import my_print

class Conf:
    def __init__(self, args=None):
        # =====
        # data
        self.train = ""
        self.dev = ""
        self.test = ""
        self.gold = ""          # gold indexes
        self.output = "output.txt"
        self.vocab = "vocab.v"
        self.vocab_rthres = 500000
        self.vocab_fthres = 0
        # =====
        # training
        #
        self.no_rebuild_vocab = False
        self.model = "models/model"
        self.reload_model = False
        self.overwrite = True
        #
        self.max_len = 100
        self.batch_size = 32
        self.shuffle_each_epoch = False
        self.rand_skip = 0.
        self.max_epochs = 50
        self.max_updates = 1000000
        self.report_freq = 1000
        self.verbose = True
        #
        self.validate_freq = -1
        self.patience = 5
        self.anneal_restarts = 5
        self.anneal_reload_best = False
        self.validate_epoch = True
        self.validate0 = False
        self.valid_metrics = ["acc", "loss"]
        #
        self.trainer_type = "adam"      # adam/sgd/momentum
        self.clip_c = 5.
        self.lrate = 0.001
        self.lrate_decay_valid = 0.99       # decay for each valid point
        self.lrate_decay_anneal = 0.75
        self.lrate_min = 0.00002
        self.momentum = 0.8
        # =====
        # tensorflow
        self.tf_device = -1
        self.tf_num_keep_ckpts = 10
        self.tf_log_device_placement = False
        self.tf_num_intra_threads = 8
        self.tf_num_inter_threads = 8
        self.tf_debug = False
        #
        if args is not None:
            self.init_from(args)

    def init_from(self, args):
        for s in args:
            fields = s.split(":")
            name, value = fields
            orig_value = self.__dict__[name]
            if orig_value is None:
                convert_f = str
            elif isinstance(orig_value, bool):
                convert_f = lambda x: bool(int(x))
            elif isinstance(orig_value, (list, tuple, dict)):
                convert_f = lambda x: json.loads(x)
            else:
                convert_f = type(orig_value)
            new_value = convert_f(value)
            my_print(f"Change conf {name}: {orig_value} -> {new_value}")
            self.__dict__[name] = new_value
