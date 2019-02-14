#

# analysis

import sys

# =====
HW1_CLASSES = ['UNK', 'Agriculture, food and drink', 'Art and architecture', 'Engineering and technology', 'Geography and places', 'History', 'Language and literature', 'Mathematics', 'Media and drama', 'Miscellaneous', 'Music', 'Natural sciences', 'Philosophy and religion', 'Social sciences and society', 'Sports and recreation', 'Video games', 'Warfare']
HW1_VOCAB = {n:i for i,n in enumerate(HW1_CLASSES)}
# =====

def main(fgold, fpred=None):
    # read them
    with open(fgold) as fd:
        golds = [line.strip() for line in fd]
    preds = None
    if fpred:
        with open(fpred) as fd:
            preds = [line.strip() for line in fd]
    #
    length = len(golds)
    if preds is not None:
        assert len(preds) == length
    # analyze them
    # 1. gold freqs
    gold_counts = {}
    for one_gold in golds:
        gold_counts[one_gold] = gold_counts.get(one_gold, 0) + 1
    # sort by freq
    sorted_classes = sorted(gold_counts.keys(), key= lambda x: -gold_counts[x])
    print(" ".join([f"{n}: {gold_counts[n]}/{gold_counts[n]/length:.4f}" for n in sorted_classes]))
    #
    corr = 0
    confusions = {z:{} for z in sorted_classes}
    if preds is not None:
        for one_gold, one_pred in zip(golds, preds):
            if one_gold == one_pred:
                corr += 1
            else:
                confusions[one_gold][one_pred] = confusions[one_gold].get(one_pred, 0) + 1
    print(f"Overall acc = {corr/length}")
    for n in sorted_classes:
        curr_dict = confusions[n]
        confusion_sorted_classes = sorted(curr_dict.keys(), key= lambda x: -curr_dict[x])
        wrong = sum(curr_dict[z] for z in confusion_sorted_classes)
        print(f"{n}({gold_counts[n]}/{gold_counts[n]/length:.4f})|wrong={wrong}")
        for c in confusion_sorted_classes:
            print(f"  ->{c}: {curr_dict[c]}/{curr_dict[c]/gold_counts[n]:.4f}")
    #
    print("=====")
    for n in sorted_classes:
        curr_dict = confusions[n]
        confusion_sorted_classes = sorted(curr_dict.keys(), key= lambda x: -curr_dict[x])
        wrong = sum(curr_dict[z] for z in confusion_sorted_classes)
        print(f"{n} & {wrong} & " + ", ".join([f"{c}({curr_dict[c]})" for c in confusion_sorted_classes]) + " \\\\")

if __name__ == '__main__':
    main(*sys.argv[1:])
    pass
