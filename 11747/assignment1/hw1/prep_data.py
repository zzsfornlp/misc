#

# scripts for preparing data into the specific format

import sys

sys.path.append("..")

from tools.utils import my_print, my_open

# =====
HW1_CLASSES = ['UNK', 'Agriculture, food and drink', 'Art and architecture', 'Engineering and technology', 'Geography and places', 'History', 'Language and literature', 'Mathematics', 'Media and drama', 'Miscellaneous', 'Music', 'Natural sciences', 'Philosophy and religion', 'Social sciences and society', 'Sports and recreation', 'Video games', 'Warfare']
HW1_VOCAB = {n:i for i,n in enumerate(HW1_CLASSES)}
# fix typo!!
HW1_VOCAB['Media and darama'] = HW1_VOCAB['Media and drama']
# =====

# format: ``label ||| text ...''
def main2(file):
    with my_open(file) as fd, my_open(file+".text", 'w') as wfd1, my_open(file+".label", 'w') as wfd2:
        # skip the first line
        # fd.readline()
        for line in fd:
            label, text = [z.strip() for z in line.rstrip().split("|||")]
            wfd1.write(text+"\n")
            wfd2.write(str(HW1_VOCAB[label])+"\n")

# split GLEU's SST2 file
# eg: python .\prep_data.py .\data\train.tsv
if __name__ == '__main__':
    main2(*sys.argv[1:])
    pass
