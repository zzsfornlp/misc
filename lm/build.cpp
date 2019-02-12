#include "build.h"
#include <cstdlib>

// other task-dependent structures

struct BuildConfig{
  // io
  string input_fname{""};   // default "" means from stdin
  string output_fname{""};  // default "" means to stdout
  string vocab_fname{"vocab.txt"};
  int order{3};             // ngram order
  string tmp_dir{"."};       // for temp files
  vector<Real> coc_fallback = {0., 0.5, 1.0, 1.5};    // if coc is 0

  static unique_ptr<BuildConfig> read(int argc, char** argv){
    unique_ptr<BuildConfig> ptr = unique_ptr<BuildConfig>(new BuildConfig());
    for(int i = 1; i < argc; i++){
      if(string(argv[i]) == "-i")
        ptr->input_fname = string(argv[++i]);
      else if(string(argv[i]) == "-o")
        ptr->output_fname = string(argv[++i]);
      else if(string(argv[i]) == "-ord")
        ptr->order = std::stoi(string(argv[++i]));
      else
        utils::zlog("Ignore unknown args.");
    }
    return ptr;
  }
};

// the procedures
vector<string> SPECIAL_TOKENS = {"<unk>", "<bos>", "<eos>"};
enum{
  IDX_UNK = 0, IDX_BOS, IDX_EOS
};
class Pipeline{
private:
  BuildConfig config;
  //
  string iname;
  std::istream* input{nullptr};
  std::ostream* output{nullptr};

  void init(){
    // input
    iname = config.input_fname;
    if(iname.size() == 0){
      iname = config.tmp_dir + PATH_SEPARATOR + "tmp_input";
      std::ofstream tmp_fout(iname);
      Helper::copy_stream(std::cin, tmp_fout);
    }
  }

  // 0. io
  void rewind_input(){
    delete input;
    input = new std::ifstream{iname};
  }

  void rewind_output(){
    if(config.output_fname.size() == 0)
      output = &std::cout;
    else{
      delete output;
      output = new std::ofstream{config.output_fname};
    }
  }

  // 2. read the (unsorted) n-grams
  unique_ptr<NgramCollection<Index>> read_ngrams(std::istream& input, const BasicVocab& vocab, Index order){
    NgramSlider<Index> slider{order, Index(IDX_BOS)};
    auto col_ptr = unique_ptr<NgramCollection<Index>>(new NgramCollection<Index>());
    //
    LineVocabStream<std::istream> streamer{input, true};
    string token{};
    while(streamer){
      streamer >> token;
      Index one = IDX_EOS;     // TODO(warn): special one as EOS
      if(token.size() > 0)
        one = vocab[token];
      col_ptr->add(slider.add(one));
      if(one==IDX_EOS)
        slider = NgramSlider<Index>{order, Index(IDX_BOS)};   // restart a new line
    }
    return col_ptr;
  }

public:
  Pipeline(BuildConfig* cc){ if(cc){ config = *cc; } init();}

  void run(){
    // step1: get vocab
    utils::zlog("Step1: get vocab");
    rewind_input();
    auto vocab = unique_ptr<BasicVocab>(new BasicVocab{*input, SPECIAL_TOKENS, {}});
    std::ofstream fout(config.vocab_fname);
    vocab->write(fout);
    fout.close();
    // step2: read again and get n-grams
    utils::zlog("Step2: read in all the ngrams");
    rewind_input();
    auto col = read_ngrams(*input, *vocab, config.order);
    // step3: sort on SuffixOrder and obtain adjusted counts by scanning
    utils::zlog("Step3: setup adjusted counts");
    auto col_counts = LMBuilder<Index>::count_on_ngrams(*col, IDX_BOS, config.coc_fallback);
    col.reset();  // release memory
    // step4: sort on ContextOrder and build lm and interpolate
    utils::zlog("Step4: build the LM");
    auto lm = LMBuilder<Index>::build_lm(*col_counts);
    lm->interpolate(IDX_UNK, IDX_BOS);
    // step5: write out
    utils::zlog("Step5: write the LM");
    rewind_output();
    lm->write_arpa(*output, *vocab);
  }
};

int main_lm(int argc, char** argv){
  auto cc = BuildConfig::read(argc, argv);
  Pipeline(cc.get()).run();
  return 0;
}

/*
TODO: special details:
1. UNK: how to deal with unk, context and unkown contex?
2. BOS: how about the starting part of a sentence?
Current solutions:
1. nope, 1-gram backoff to uniform distri., including UNK.
2. Padding to the highest order and special counting for them.
-----
-- query: how to deal with Unknown Context
-- BOS/UNK count+=2
-- remove padding
*/

/*
BUGS: "unanimity," -> seems to be solved by fixed Streamer>> -> caused by empty line
*/

//#define RUN_LM
#ifdef RUN_LM
int main(int argc, char** argv){
  main_lm(argc, argv);
  return 0;
}
#endif // RUN_LM
