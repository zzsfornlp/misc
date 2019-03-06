#

import gzip, bz2, sys, logging
import pickle

def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    elif filename.endswith('.bz2'):
        return bz2.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)

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

def zlog(s, name="main"):
    logger = Logger.get_logger(name)
    logger.info(s)

def printd(d):
    for n in sorted(d.keys()):
        zlog("-- %s: %s" % (n, d[n]))

# save and load
class PickleRW(object):
    @staticmethod
    def read(fname):
        with open(fname, "rb") as fd:
            x = pickle.load(fd)
        zlog(f"Read {str(x)} from {fname}.")
        return x

    @staticmethod
    def write(fname, x):
        with open(fname, "wb") as fd:
            pickle.dump(x, fd)
        zlog(f"Write {str(x)} to {fname}.")

