## implementation of text classifier

This is the implementation of assignment 1 for the course nn4nlp.

The repo:
* `tools/*`: Tools for data-loading, training and other utils.
* `hw1/hw1.py`: codes for this assignment.
* `hw1/prep_*.py`: preparing and analyzing scripts.

------

* notes of some of the sources

Most of the codes in this repo is implemented from scratch, especially the model part (mainly in `hw1/hw1.py`, `tools\nntf.py`), which is implemented with Tensorflow. (with Nematus and tensorflow/nmt as references, but surely no directly re-used codes).

Some of the codes in `tools` are directly adopted (some with small modifications) from one of my previous projects `znmt-merge` ([link](https://github.com/zzsfornlp/znmt-merge)). For example, some of the helper functions in `tools/utils` like `Timer` or `shuffle`, the main model training manager class `tools/run.py:TrainingProgress,Runner`. Please inform me if these parts of the codes violate the assignment rules and I will make modifications correspondingly.
