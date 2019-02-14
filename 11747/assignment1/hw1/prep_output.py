#

import sys

HW1_CLASSES = ['UNK', 'Agriculture, food and drink', 'Art and architecture', 'Engineering and technology', 'Geography and places', 'History', 'Language and literature', 'Mathematics', 'Media and drama', 'Miscellaneous', 'Music', 'Natural sciences', 'Philosophy and religion', 'Social sciences and society', 'Sports and recreation', 'Video games', 'Warfare']

#
def main(file_in, file_out):
    with open(file_in) as fin, open(file_out, 'w') as fout:
        ints = [HW1_CLASSES[int(line.strip())] for line in fin]
        fout.write("\n".join(ints))
        # for line in fin:
        #     x = int(line.strip())
        #     fout.write(f"{HW1_CLASSES[x]}\n")

if __name__ == '__main__':
    main(*sys.argv[1:])
    pass
