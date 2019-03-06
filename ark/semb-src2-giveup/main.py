#

import sys
sys.path.append("../")
from src.dataloader import *
from src.evaluations import *
from src.args import add_args

from src.zmodel import syntax_skigram as zmodel
import pdb

from src2.model2 import model2

#
def main():
    args = add_args()
    print(args)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    args.cuda = torch.cuda.is_available()
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)
    # device = torch.device('cuda') if args.cuda else torch.device("cpu")
    device = torch.device("cpu")        # currently always cpu for torch
    run(args, device)

def run(args, device):
    dataset = DataSet(args, args.in_path)
    # ===== with new model

    data_iterator = lazy_file_iterator(dataset, args, device, args.checkpoint_interval, args.save_checkpoint_interval)
    # if args.use_zmodel:
    #     model = zmodel(args, dataset, data_iterator)
    # else:
    #     model = syntax_skigram(args, dataset, data_iterator)
    # =====
    # model.init(device)

    model = model2(args, dataset, data_iterator)

    training_stats = {"pos_h_loss": [], "pos_m_loss": [], "neg_h_loss": [], "neg_m_loss": [], "loss": [], "bad_gradients": []}
    training_stats_more = {"pos_h_loss": [], "pos_m_loss": [], "neg_h_loss": [], "neg_m_loss": [], "loss": [], "bad_gradients": []}
    n_words_proc = 0
    tic = time.time()

    best_sim_avg_acc = 0.
    for datum in data_iterator:
        n_words_proc += len(datum[4])

        train_steps = data_iterator.tot_batches

        model.step(datum, training_stats, return_loss=(train_steps % args.return_loss_steps == 0))

        if train_steps % args.display_steps == 0:
            ss = 'Step=%i, %i words/s' % (train_steps, int(n_words_proc / (time.time() - tic)))
            lr = model.cur_lr
            ss += ", lr=%.6f" % lr
            for k, v in training_stats.items():
                if type(v) != list or len(v) == 0:
                    continue
                mean_value = np.mean(v)
                training_stats_more[k].append(mean_value)
                overall_mean_value = np.mean(training_stats_more[k])
                ss += ", %s=%.4f(%.4f)" % (k, mean_value, overall_mean_value)
            ss += ", progress=%.4f" % data_iterator.progress
            print(ss)
            for k, v in training_stats.items():
                if type(training_stats[k]) == list:
                    del training_stats[k][:]

            if train_steps % 1000 == 0:
                n_words_proc = 0
                tic = time.time()
                #
                if train_steps % 100000 == 0:
                    training_stats_more = {"pos_h_loss": [], "pos_m_loss": [], "neg_h_loss": [], "neg_m_loss": [],
                                           "loss": [], "bad_gradients": []}
                    print("Clear overall stats!")

        if data_iterator.evalute:
            sim_avg_acc = model.evaluate()
            if sim_avg_acc is not None and sim_avg_acc > best_sim_avg_acc:
                best_sim_avg_acc = sim_avg_acc
                if data_iterator.progress > 0.2:
                    model.save(save_path_suffix="best")
            data_iterator.evalute = False

        if data_iterator.progress > 0.2 and data_iterator.save:
            model.save(save_path_suffix=("%.2f" % data_iterator.progress))
            data_iterator.save = False

        if model.cur_lr < args.min_lr:
            print(f"Learning rate <= {args.min_lr}, stop training!")
            model.save(save_path_suffix=("%.2f" % data_iterator.progress))
            break

#
if __name__ == '__main__':
    main()
