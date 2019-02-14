#

#
import logging, sys
import gzip, bz2
import os, subprocess
import numpy as np
import time

class Logger:
    cached_loggers = {}
    @staticmethod
    def get_logger(name, level=logging.INFO, handler=sys.stderr,
                   formatter='%(asctime)s - %(name)s - %(levelname)s - %(message)s'):
        if name not in Logger.cached_loggers:
            logger = logging.getLogger(name)
            logger.setLevel(logging.INFO)
            formatter = logging.Formatter(formatter)
            stream_handler = logging.StreamHandler(handler)
            stream_handler.setLevel(level)
            stream_handler.setFormatter(formatter)
            logger.addHandler(stream_handler)
            Logger.cached_loggers[name] = logger
        return Logger.cached_loggers[name]

# =====
def my_print(s, lname="main"):
    Logger.get_logger(lname).info(s)

def my_open(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

def my_system(cmd, popen=False, print=False, check=False):
    if print:
        my_print("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = p.stdout.read()
        if print:
            my_print("Output is: %s" % output)
    else:
        n = os.system(cmd)
        output = None
    if check:
        assert n==0
    return output

# must provide parallel data of one line per instance
def shuffle_files(files, output_suffix=".shuf"):
    # shuffle in memory!
    my_print(f"Shuffling files: {files}, reading ...")
    fds = [my_open(f) for f in files]
    instances = []
    while True:
        lines = [fd.readline() for fd in fds]
        is_eos = [t=="" for t in lines]
        if all(is_eos):
            break
        assert not any(is_eos), f"Unmatched data, is_eos={['%s:%s'%(f,z) for f, z in zip(files, is_eos)]}"
        instances.append(lines)
    for fd in fds:
        fd.close()
    # shuffle
    my_print(f"Shuffling files: {files}, shuffling ...")
    np.random.shuffle(instances)
    # writing
    my_print(f"Shuffling files: {files}, writing with suffix of {output_suffix} ...")
    output_names = [f+output_suffix for f in files]
    wfds = [my_open(f, 'w') for f in output_names]
    for inst in instances:
        assert len(wfds) == len(wfds)
        for idx, line in enumerate(inst):
            wfds[idx].write(line)
    for fd in wfds:
        fd.close()
    return output_names

# =====
class Timer:
    START = 0.

    @staticmethod
    def init():
        Timer.START = time.time()

    @staticmethod
    def systime():
        return time.time()-Timer.START

    def __init__(self, tag, info="", print_date=False, quiet=False):
        self.tag = tag
        self.print_date = print_date
        self.quiet = quiet
        self.info = info
        self.accu = 0.   # accumulated time
        self.paused = False
        self.start = Timer.systime()

    def pause(self):
        if not self.paused:
            cur = Timer.systime()
            self.accu += cur - self.start
            self.start = cur
            self.paused = True

    def resume(self):
        if not self.paused:
            my_print("WARN: Timer should be paused to be resumed.")
        else:
            self.start = Timer.systime()
            self.paused = False

    def get_time(self):
        self.pause()
        self.resume()
        return self.accu

    def init_state(self):
        return 0.

    def begin(self):
        self.start = Timer.systime()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            my_print("Start timer %s: %s at %.3f. (%s)" % (self.tag, self.info, self.start, cur_date))

    def end(self, s=None):
        self.pause()
        if not self.quiet:
            cur_date = time.ctime() if self.print_date and not self.quiet else ""
            my_print("End timer %s: %s at %.3f, the period is %.3f seconds. (%s)" % (self.tag, self.info, Timer.systime(), self.accu, cur_date))
        # accumulate
        if s is not None:
            return s+self.accu
        else:
            return None

    def __enter__(self):
        self.begin()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end()
