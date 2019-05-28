#

# prepare data and embeddings

# add the bigger ones
# 'ar bg ca zh hr cs da nl en et fi fr de he hi id it ja ko la lv no pl pt ro ru sk sl es sv uk'
#print("\n".join([z[1][0].split("-")[0].split("_")[1] for z in x]))
LANGUAGE_LIST = (
    # ["ar", ["UD_Arabic-PADT"], "Afro-Asiatic.Semitic"],
    # ["bg", ["UD_Bulgarian-BTB"], "IE.Slavic.South"],
    # ["ca", ["UD_Catalan-AnCora"], "IE.Romance.West"],
    # ["zh", ["UD_Chinese-GSD"], "Sino-Tibetan"],
    # ["hr", ["UD_Croatian-SET"], "IE.Slavic.South"],
    # ["cs", ["UD_Czech-PDT", "UD_Czech-CAC", "UD_Czech-CLTT", "UD_Czech-FicTree"], "IE.Slavic.West"],
    # ["da", ["UD_Danish-DDT"], "IE.Germanic.North"],
    # ["nl", ["UD_Dutch-Alpino", "UD_Dutch-LassySmall"], "IE.Germanic.West"],
    # ["en", ["UD_English-EWT"], "IE.Germanic.West"],
    # ["et", ["UD_Estonian-EDT"], "Uralic.Finnic"],
    # ["fi", ["UD_Finnish-TDT"], "Uralic.Finnic"],
    # ["fr", ["UD_French-GSD"], "IE.Romance.West"],
    # ["de", ["UD_German-GSD"], "IE.Germanic.West"],
    # ["he", ["UD_Hebrew-HTB"], "Afro-Asiatic.Semitic"],
    # ["hi", ["UD_Hindi-HDTB"], "IE.Indic"],
    # ["id", ["UD_Indonesian-GSD"], "Austronesian.Malayo-Sumbawan"],
    # ["it", ["UD_Italian-ISDT"], "IE.Romance.Italo"],
    # ["ja", ["UD_Japanese-GSD"], "Japanese"],
    # ["ko", ["UD_Korean-GSD", "UD_Korean-Kaist"], "Korean"],
    # ["la", ["UD_Latin-PROIEL"], "IE.Latin"],
    # ["lv", ["UD_Latvian-LVTB"], "IE.Baltic"],
    # ["no", ["UD_Norwegian-Bokmaal", "UD_Norwegian-Nynorsk"], "IE.Germanic.North"],
    # ["pl", ["UD_Polish-LFG", "UD_Polish-SZ"], "IE.Slavic.West"],
    # ["pt", ["UD_Portuguese-Bosque", "UD_Portuguese-GSD"], "IE.Romance.West"],
    # ["ro", ["UD_Romanian-RRT"], "IE.Romance.East"],
    # ["ru", ["UD_Russian-SynTagRus"], "IE.Slavic.East"],
    # ["sk", ["UD_Slovak-SNK"], "IE.Slavic.West"],
    # ["sl", ["UD_Slovenian-SSJ", "UD_Slovenian-SST"], "IE.Slavic.South"],
    # ["es", ["UD_Spanish-GSD", "UD_Spanish-AnCora"], "IE.Romance.West"],
    # ["sv", ["UD_Swedish-Talbanken"], "IE.Germanic.North"],
    # ["uk", ["UD_Ukrainian-IU"], "IE.Slavic.East"],
    ["fa", ["UD_Persian-Seraji"], "?"],
    ["ur", ["UD_Urdu-UDTB"], "?"],
    ["tr", ["UD_Turkish-IMST"], "?"],
)

TRAIN_LANG = "en"

# confs
UD2_DIR = "../../data/ud-treebanks-v2.2/"
OUT_DIR = "./data/"
LIB_DIR = "./data/fastText_multilingual/"

# ===== help
import os, subprocess, sys, gzip

sys.path.append(LIB_DIR)        # project embeddings

from fasttext import FastVector

printing = lambda x: print(x, file=sys.stderr, flush=True)

def system(cmd, pp=False, ass=False, popen=False):
    if pp:
        printing("Executing cmd: %s" % cmd)
    if popen:
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        n = p.wait()
        output = str(p.stdout.read().decode())
    else:
        n = os.system(cmd)
        output = None
    if pp:
        printing("Output is: %s" % output)
    if ass:
        assert n==0
    return output

def zopen(filename, mode='r', encoding="utf-8"):
    if filename.endswith('.gz'):
        # "t" for text mode of gzip
        return gzip.open(filename, mode+"t", encoding=encoding)
    else:
        return open(filename, mode, encoding=encoding)
# =====

#
def deal_conll_file(fin, fout):
    for line in fin:
        line = line.strip()
        fields = line.split("\t")
        if len(line) == 0:
            fout.write("\n")
        else:
            try:
                z = int(fields[0])
                fields[4] = fields[3]
                fields[3] = "_"
                fout.write("\t".join(fields)+"\n")
            except:
                pass

#
def main():
    # first get the English one
    lang = "en"
    system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
    # en_dict = FastVector(vector_file='%s/wiki.en.vec' % OUT_DIR)
    for zzz in LANGUAGE_LIST:
        lang, fnames = zzz[0], zzz[1]
        printing("Dealing with lang %s." % lang)
        for curf in ["train", "dev", "test"]:
            out_fname = "%s/%s_%s.conllu" % (OUT_DIR, lang, curf)
            fout = zopen(out_fname, "w")
            for fname in fnames:
                last_name = fname.split("-")[-1].lower()
                path_name = "%s/%s/%s_%s-ud-%s.conllu" % (UD2_DIR, fname, lang, last_name, curf)
                if os.path.exists(path_name):
                    with zopen(path_name) as fin:
                        deal_conll_file(fin, fout)
            fout.close()
            # stat
            system('cat %s | grep -E "^$" | wc' % out_fname, pp=True)
            system('cat %s | grep -Ev "^$" | wc' % out_fname, pp=True)
            system("cat %s | grep -Ev '^$' | cut -f 5 -d $'\t'| grep -Ev 'PUNCT|SYM' | wc" % out_fname, pp=True)
        # get original embed
        system("wget -nc -O %s/wiki.%s.vec https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki.%s.vec" % (OUT_DIR, lang, lang), pp=True)
        # project with LIB-matrix
        lang_dict = FastVector(vector_file='%s/wiki.%s.vec' % (OUT_DIR, lang))
        lang_dict.apply_transform("%s/alignment_matrices/%s.txt" % (LIB_DIR, lang))
        lang_dict.export("%s/wiki.multi.%s.vec" % (OUT_DIR, lang))

if __name__ == '__main__':
    main()

# =====
def deal_conll_file(fin, fout, replace):
    for line in fin:
        line = line.strip()
        fields = line.split("\t")
        if len(line) == 0:
            fout.write("\n")
        else:
            try:
                z = int(fields[0])
                fields[4] = fields[3]
                fields[3] = "_"
                if replace:
                    fields[6] = fields[9]
                fout.write("\t".join(fields)+"\n")
            except:
                pass

def f2_main(infile):
    with open(infile) as fd, open(infile+".gold", 'w') as gfd:
        deal_conll_file(fd, gfd, replace=False)
    with open(infile) as fd, open(infile + ".pred", 'w') as gfd:
        deal_conll_file(fd, gfd, replace=True)

#
def f2_main2(fin, fout):
    prev_tag = None
    prev_perc = ""
    for line in fin:
        fields = line.split()
        assert len(fields) == 4
        cur_tag = fields[1]
        line_uas = fin.readline()
        line_perc = fin.readline()
        cur_uas = line_uas.split()[1]
        cur_perc = line_perc.split()[1]
        if cur_tag == prev_tag:
            # assert float(cur_perc)-float(prev_perc)<=0.001
            fout.write(f"\t{cur_uas}")
        else:
            prev_tag = cur_tag
            prev_perc = cur_perc
            fout.write(f"\n{cur_tag}\t{cur_perc}\t{cur_uas}")

# if __name__ == '__main__':
#     # main(*sys.argv[1:])
#     main2(sys.stdin, sys.stdout)

# for f in *.conllu; do python3 deal.py $f; done
# then manually delete long ar sentences

# =====


# step 1:
# setup the data
# python3 -u get_extra.py |& grep -v "s$" | tee data/log

# step 2:
# decode
"""
for model_id in {1..5}; do
for cur_lang in fa ur tr;do
PYTHONPATH=../src/ CUDA_VISIBLE_DEVICES=2 python2 -u ../src/examples/analyze.py --parser biaffine --ordered --gpu \
--punctuation 'PUNCT' 'SYM' --out_filename analyzer.$cur_lang.$which_set.out${model_id} --model_name 'network.pt' \
--test "./data/${cur_lang}_test.conllu" --model_path ../final_gtrans.sh_${model_id}/model/ --extra_embed "./data/wiki.multi.${cur_lang}.vec"
done
done
#
for cl in fa ur tr; do echo; echo "CURRENT ${cl}"; python3 eval.py "1" 0 final_gtrans.sh_1/model/analyzer.${cl}.test.out_gold final_gtrans.sh_*/model/analyzer.${cl}.test.out_pred; done
"""

# error breakdown in LABEL
"""
for cl in ja ar id; do
# overall score
echo "ZZ all LEX ${cl}"; python3 eval.py "1" 0 ../../final_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ all DELEX ${cl}"; python3 eval.py "1" 0 ../../final_delex_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_delex_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ all START ${cl}"; python3 eval.py "1" 0 ${cl}_parse_pred_start.conllu.gold ${cl}_parse_pred_start.conllu.pred;
echo "ZZ all END ${cl}"; python3 eval.py "1" 0 ${cl}_parse_pred_end.conllu.gold ${cl}_parse_pred_end.conllu.pred;
#
for ll in case nmod amod det obl nsubj root advmod conj obj cc mark aux acl nummod flat cop advcl xcomp appos compound expl ccomp fixed iobj parataxis dep csubj orphan discourse clf goeswith vocative list dislocated reparandum; do
echo "ZZ ${ll} LEX ${cl}"; python3 eval.py "label=='${ll}'" 0 ../../final_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ ${ll} DELEX ${cl}"; python3 eval.py "label=='${ll}'" 0 ../../final_delex_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_delex_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ ${ll} START ${cl}"; python3 eval.py "label=='${ll}'" 0 ${cl}_parse_pred_start.conllu.gold ${cl}_parse_pred_start.conllu.pred;
echo "ZZ ${ll} END ${cl}"; python3 eval.py "label=='${ll}'" 0 ${cl}_parse_pred_end.conllu.gold ${cl}_parse_pred_end.conllu.pred;
done
done
"""

# error breakdown in DepDist
"""
for cl in ja ar id; do
# overall score
echo "ZZ all LEX ${cl}"; python3 eval.py "1" 0 ../../final_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ all DELEX ${cl}"; python3 eval.py "1" 0 ../../final_delex_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_delex_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ all START ${cl}"; python3 eval.py "1" 0 ${cl}_parse_pred_start.conllu.gold ${cl}_parse_pred_start.conllu.pred;
echo "ZZ all END ${cl}"; python3 eval.py "1" 0 ${cl}_parse_pred_end.conllu.gold ${cl}_parse_pred_end.conllu.pred;
#
for dd in "ddist<-10" "ddist==-10" "ddist==-9" "ddist==-8" "ddist==-7" "ddist==-6" "ddist==-5" "ddist==-4" "ddist==-3" "ddist==-2" "ddist==-1" "ddist==1" "ddist==2" "ddist==3" "ddist==4" "ddist==5" "ddist==6" "ddist==7" "ddist==8" "ddist==9" "ddist==10" "ddist>10" "p_ddist<-10" "p_ddist==-10" "p_ddist==-9" "p_ddist==-8" "p_ddist==-7" "p_ddist==-6" "p_ddist==-5" "p_ddist==-4" "p_ddist==-3" "p_ddist==-2" "p_ddist==-1" "p_ddist==1" "p_ddist==2" "p_ddist==3" "p_ddist==4" "p_ddist==5" "p_ddist==6" "p_ddist==7" "p_ddist==8" "p_ddist==9" "p_ddist==10" "p_ddist>10"; do
echo "ZZ ${dd} LEX ${cl}"; python3 eval.py "${dd}" 0 ../../final_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ ${dd} DELEX ${cl}"; python3 eval.py "${dd}" 0 ../../final_delex_gtrans.sh_1/model/analyzer.${cl}.test.out_gold ../../final_delex_gtrans.sh_*/model/analyzer.${cl}.test.out_pred;
echo "ZZ ${dd} START ${cl}"; python3 eval.py "${dd}" 0 ${cl}_parse_pred_start.conllu.gold ${cl}_parse_pred_start.conllu.pred;
echo "ZZ ${dd} END ${cl}"; python3 eval.py "${dd}" 0 ${cl}_parse_pred_end.conllu.gold ${cl}_parse_pred_end.conllu.pred;
done
done
"""
