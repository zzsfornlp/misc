#

# mostly adopted from znmt-merge

from .utils import my_open, my_print, my_system, Timer
from .nntf import TFMode
import json
import numpy as np

#
MIN_V = -99999.

class TrainingProgress(object):
    # for score, the larger the better
    def load_from_json(self, file_name):
        with my_open(file_name, 'r') as fd:
            self.__dict__.update(json.load(fd))

    def save_to_json(self, file_name):
        with my_open(file_name, 'w') as fd:
            json.dump(self.__dict__, fd, indent=2)

    def __init__(self, patience, anneal_restarts):
        self.bad_counter = 0
        self.bad_points = []
        self.anneal_restarts_done = 0
        self.anneal_restarts_points = []
        self.uidx = 0                   # update
        self.eidx = 0                   # epoch
        self.estop = False
        self.hist_points = []
        self.hist_scores = []            # the bigger the better
        self.hist_trains = []
        self.best_score = None
        self.best_point = None
        # states
        self.patience = patience
        self.anneal_restarts_cap = anneal_restarts
        #
        self.hists = []
        self.sortings = []

    @property
    def num_anneals(self):
        return self.anneal_restarts_done

    def record(self, ss, score, trains):
        def _getv(z):
            return MIN_V if z is None else z[0]
        # record one score, return (if-best, if-anneal)
        if_best = if_anneal = False
        self.hist_points.append(ss)
        self.hist_scores.append(score)
        self.hist_trains.append(trains)
        if _getv(score) > _getv(self.best_score):
            self.bad_counter = 0
            self.best_score = score
            self.best_point = ss
            if_best = True
        else:
            self.bad_counter += 1
            self.bad_points.append([ss, self.best_score, self.best_point])      # bad_point, .best-at-that-time
            my_print("Bad++, now bad/anneal is %s/%s." % (self.bad_counter, self.anneal_restarts_done))
            if self.bad_counter >= self.patience:
                self.bad_counter = 0
                self.anneal_restarts_points.append([ss, self.best_score, self.best_point])
                if self.anneal_restarts_done < self.anneal_restarts_cap:
                    self.anneal_restarts_done += 1
                    my_print("Anneal plus one, now %s." % (self.anneal_restarts_done,))
                    if_anneal = True
                else:
                    my_print("WARN: Sorry, Early Update !!")
                    self.estop = True
        #
        self.hists = [(_1, _getv(_2)) for _1,_2 in zip(self.hist_points, self.hist_scores)]
        self.sortings = sorted(self.hists, key=lambda x: x[-1], reverse=True)[:10]
        return if_best, if_anneal

    def report(self):
        def _getv(z):
            return MIN_V if z is None else z[0]
        for k in sorted(self.__dict__):
            my_print("Training progress results: %s = %s." % (k, self.__dict__[k]))

    def link_bests(self, basename):
        my_system("rm ztop_model*")
        for i, pair in enumerate(self.sortings):
            my_system("ln -s %s%s ztop_model%s" % (basename, pair[0], i), print=True)

# class ValidScore(object):
#     pass

class Runner(object):
    TP_TAIL = ".progress.json"
    TR_TAIL = ".trainer.shadow"
    CURR_SUFFIX = ".curr.pkl"
    BEST_SUFFIX = ".best.pkl"

    def __init__(self, args, model):
        # Used options: lrate, moment, trainer_type, clip_c, max_epochs, max_updates, overwrite
        # --: anneal_reload_best, anneal_restarts, patience
        opts = args.__dict__.copy()
        #
        self.opts = opts
        self._tp = TrainingProgress(opts["patience"], opts["anneal_restarts"])
        self._mm = model
        # learning rate schedule
        self.lrate_decay_valid = opts["lrate_decay_valid"]
        self.lrate_decay_anneal = opts["lrate_decay_anneal"]
        self.start_lrate = opts["lrate"]
        self.lrate_min = opts["lrate_min"]
        self.cur_lrate = self.start_lrate

    # load and save
    def load(self, basename, load_process):
        # load model
        self._mm.load(basename)
        my_print("Reload model from %s." % basename)
        # load progress
        if load_process:
            tp_name = basename + Runner.TP_TAIL
            tr_name = basename + Runner.TR_TAIL
            my_print("Reload trainer from %s and %s." % (tp_name, tr_name))
            self._tp.load_from_json(tp_name)

    def save(self, basename):
        # save model
        self._mm.save(basename)
        # save progress
        tp_name = basename + Runner.TP_TAIL
        tr_name = basename + Runner.TR_TAIL
        my_print("Save trainer to %s and %s." % (tp_name, tr_name))
        self._tp.save_to_json(tp_name)

    # helpers
    def _finished(self):
        return self._tp.estop or self._tp.eidx >= self.opts["max_epochs"] \
                or self._tp.uidx >= self.opts["max_updates"]

    # ========
    # todo(warn): template methods, to be implemented
    def _validate_them(self, dev_iter, metrics):
        raise NotImplementedError("Not here for basic trainer!")

    def _test_them(self, test_iter, output_name):
        raise NotImplementedError("Not here for basic trainer!")

    def _get_recorder(self, name):
        raise NotImplementedError("Not here for basic trainer!")

    # todo(warn): combined fb and update
    def _fb_once(self, insts, cur_lrate):
        raise NotImplementedError("Not here for basic trainer!")
    # ========

    # main rountines
    def _validate(self, dev_iter, name=None, training_states=None):
        # validate and log in the stats
        ss = ".e%s-u%s" % (self._tp.eidx, self._tp.uidx) if name is None else name
        with Timer(tag="valid", info="Valid %s" % ss, print_date=True):
            # checkpoint - write current
            curr_ckpt = self.opts["model"]+Runner.CURR_SUFFIX
            self.save(curr_ckpt)
            if not self.opts["overwrite"]:
                self.save(self.opts["model"]+ss)
            # load current model before evaluate
            self._mm.load(curr_ckpt, mode=TFMode.EVAL)
            self._mm.load(curr_ckpt, mode=TFMode.INFER)
            # validate
            score = self._validate_them(dev_iter, self.opts["valid_metrics"])
            my_print("Validating %s for %s: score is %s." % (self.opts["valid_metrics"], ss, score))
            # write best and update stats
            if_best, if_anneal = self._tp.record(ss, score, training_states)
            if if_best:
                self.save(self.opts["model"]+Runner.BEST_SUFFIX)
            if if_anneal:
                if self.opts["anneal_reload_best"]:
                    self.load(self.opts["model"]+Runner.BEST_SUFFIX, False)   # load model, but not process
                # update lrate
                self.cur_lrate *= self.lrate_decay_anneal
                my_print(f"Decay(anneal) lrate to {self.cur_lrate}")
            self.cur_lrate *= self.lrate_decay_valid
            my_print(f"Decay(valid) lrate to {self.cur_lrate}")
            self.cur_lrate = max(self.cur_lrate, self.lrate_min)
            my_print(f"Cur-lrate = {self.cur_lrate}")
        self._tp.report()
        # self._tp.link_bests(self.opts["model"])
        my_print("")     # to see it more clearly

    # main training
    def train(self, train_iter, dev_iter):
        one_recorder = self._get_recorder("CHECK")
        if self.opts["validate0"]:
            self._validate(dev_iter, training_states=one_recorder.state())      # validate once before training
        while not self._finished():     # epochs
            # utils.printing("", func="info")
            with Timer(tag="Train-Iter", info="Iter %s" % self._tp.eidx, print_date=True) as et:
                iter_recorder = self._get_recorder("ITER-%s" % self._tp.eidx)
                for insts in train_iter:
                    # todo(warn): end-of-epoch mark!
                    if insts is None:
                        break
                    if np.random.random_sample() < self.opts["rand_skip"]:     # introduce certain randomness
                        continue
                    # training for one batch
                    loss = self._fb_once(insts, self.cur_lrate)
                    self._tp.uidx += 1
                    one_recorder.record(insts, loss, 1)
                    iter_recorder.record(insts, loss, 1)
                    if self.opts["verbose"] and self._tp.uidx % self.opts["report_freq"] == 0:
                        one_recorder.report("Training process: ")
                    # time to validate and save best model ??
                    if self.opts["validate_freq"]>0 and self._tp.uidx % self.opts["validate_freq"] == 0:
                        one_recorder.report()
                        self._validate(dev_iter, training_states=one_recorder.state())
                        one_recorder.reset()
                        if self._finished():
                            break
                iter_recorder.report()
                if self.opts["validate_epoch"]:
                    # here, also record one_recorder, might not be accurate
                    one_recorder.report()
                    self._validate(dev_iter, name=".ev%s" % self._tp.eidx, training_states=one_recorder.state())
                    one_recorder.reset()
                self._tp.eidx += 1

    #
    def test(self, test_iter):
        model_name = self.opts["model"]+Runner.BEST_SUFFIX
        # load current model before evaluate
        self._mm.load(model_name, mode=TFMode.EVAL)
        self._mm.load(model_name, mode=TFMode.INFER)
        with Timer(tag="test", info="Testing", print_date=True):
            self._test_them(test_iter, self.opts["output"])

#
class StatusRecorder:
    def state(self):
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def report(self):
        raise NotImplementedError()

    def record(self, *args, **kwargs):
        raise NotImplementedError()
