#

import argparse

import torch
from torch.autograd import Variable

import data

parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')

# Model parameters.
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN)')
parser.add_argument('--checkpoint', '--save', type=str, default='./model.pt',
                    help='model checkpoint to use')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--words', type=int, default='1000',
                    help='number of words to generate')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--temperature', type=float, default=1.0,
                    help='temperature - higher will increase diversity')
parser.add_argument('--log-interval', type=int, default=100,
                    help='reporting interval')
# =====
parser.add_argument("--corr_type", type=str, default="word")  # word, char, bpe
# =====
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

if args.temperature < 1e-3:
    parser.error("--temperature has to be greater or equal 1e-3")

with open(args.checkpoint, 'rb') as f:
    if args.cuda:
        model = torch.load(f)
    else:
        model = torch.load(f, map_location='cpu')
model.eval()
if args.model == 'QRNN':
    model.reset()

if args.cuda:
    model.cuda()
else:
    model.cpu()

corpus = data.Corpus(args.data)
ntokens = len(corpus.dictionary)

lines = []
with open(args.data + 'test.txt', 'r') as test:
    lines = test.readlines()
lines = [line for line in lines if line.strip() != '']
first_line = lines[0].split() + ['<eos>']
first_word = first_line[0]

# input = Variable(torch.rand(1, 1).mul(ntokens).long(), volatile=True)
input = torch.tensor([[corpus.dictionary.word2idx[first_word]]])
if args.cuda:
    input.data = input.data.cuda()
hidden = model.init_hidden(1)

success = 0
error = 0
wpa = 0

# =====
word2idx = corpus.dictionary.word2idx
correctness = [1]
all_token_strs = []

for line in lines:
    tokens = line.split() + ['<eos>']
    all_token_strs.extend(tokens)

# feed idx and predict idx+1
for idx in range(len(all_token_strs)-1):
    feed_idx = word2idx[all_token_strs[idx]]
    gold_idx = word2idx[all_token_strs[idx+1]]
    input.data.fill_(feed_idx)
    output, hidden = model(input, hidden)
    _, argmax_ids = output.max(-1)
    pred_idx = argmax_ids.numpy().item()
    correctness.append(int(gold_idx==pred_idx))

# =====
def my_eval(correctness, orig_words, corr_type):
    if corr_type == "word":
        end_of_words = [1] * len(correctness)  # every word is an end of word
    elif corr_type == "char":
        # using "_" as breaker
        # using "_" as breaker
        end_of_words = [int(x == "_" or x == "<eos>" or (idx < len(orig_words) - 1 and orig_words[idx + 1] == "<eos>"))
                        for idx, x in enumerate(orig_words)]
    else:
        # using "@@" as connectors
        end_of_words = [int(not x.endswith("@@")) for x in orig_words]
    end_of_words[-1] = True  # force the ending
    # evaluate
    orig_count, word_count = len(end_of_words), sum(end_of_words)
    orig_corr_count, word_corr_count = 0, 0
    cont_corr = True
    for cur_corr, cur_eow in zip(correctness, end_of_words):
        cur_corr = bool(cur_corr)
        if cur_corr:
            orig_corr_count += 1
        cont_corr = (cont_corr and cur_corr)
        if cur_eow:
            if cont_corr:
                word_corr_count += 1
            cont_corr = True
    orig_acc, word_acc = orig_corr_count/orig_count, word_corr_count/word_count
    print("Orig: %s/%s=%s, Word: %s/%s=%s" % (orig_corr_count, orig_count, orig_acc, word_corr_count, word_count, word_acc))
# =====

my_eval(correctness, all_token_strs, args.corr_type)

