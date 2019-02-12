#ifndef _H_LM_HELP
#define _H_LM_HELP

// A simple Mod-K&N-LM implementation, KenLM as the reference

#include <vector>
#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <memory>
#include <algorithm>
#include <queue>
#include <iostream>
#include <iterator>
#include <cmath>
#include <cstdio>

using std::unordered_map;
using std::string;
using std::vector;
using std::unique_ptr;

// types
using CountInt = unsigned;
using Index = unsigned;
using SignedIndex = int;
using Real = float;

#if defined(WIN32) || defined(_WIN32) 
#define PATH_SEPARATOR "\\" 
#else 
#define PATH_SEPARATOR "/" 
#endif 

namespace utils{
  void zcheck(bool e, const string& err){
    if(!e) throw err;
  }

  void zlog(const string& s){
    std::cerr << s << std::endl;
  }

  // from https://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
  template<typename ... Args>
  string string_format(const std::string& format, Args ... args)
  {
    size_t size = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
    unique_ptr<char[]> buf(new char[size]);
    std::snprintf(buf.get(), size, format.c_str(), args ...);
    return string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
  }
};

// helpers
struct Helper{
  // streamer
  template<class T>
  class Stream{
  public:
    virtual Stream& operator>> (T&) = 0;
    virtual operator bool() = 0;
  };

  // obtain freq map
  template<class KeyType, class CountType>
  static unique_ptr<unordered_map<KeyType, CountInt>> count_freq(Stream<KeyType>& input){
    typedef unordered_map<KeyType, CountInt> VocabMap;
    // read them once
    unique_ptr<VocabMap> ptr{new VocabMap{}};
    KeyType token{};
    while(input){
      input >> token;
      auto item_ptr = ptr->find(token);
      if(item_ptr == ptr->end())
        ptr->insert({token, 1});
      else
        item_ptr->second++;
    }
    return ptr;
  }

  // rank highest freq of a dictionary, return ranked keys
  template<class KeyType, class CountType>
  static unique_ptr<vector<KeyType>> rank_key(const unordered_map<KeyType, CountType>& freq_map){
    typedef vector<KeyType> VocabList;
    //
    unique_ptr<VocabList> ptr{new VocabList{}};
    for(const auto& tok : freq_map)
      ptr->push_back(tok.first);
    // sort with frequency
    std::sort(ptr->begin(), ptr->end(), 
      [&freq_map](const KeyType& a, const KeyType& b){ return freq_map.at(a) > freq_map.at(b); });
    return ptr;
  }

  // copy stream
  static void copy_stream(std::istream& input, std::ostream& output){
    output << input.rdbuf();
  }

  // log
#define LOG_MIN -99.0
  static Real log(const Real x){ 
    if(x <= 0) return LOG_MIN;
    return std::log10(x); 
  }
#undef LOG_MIN
  static Real exp(const Real x){ return std::pow(10, x); }
  static Real logsumexp(const vector<Real>& vs){
    Real maxv = *(std::max_element(vs.begin(), vs.end()));
    Real accu = 0;
    for(Real v : vs)
      accu += exp(v-maxv);
    return maxv + log(accu);
  }
};

// reader
// read tokens from input-stream
template<class SrcStream>
class LineVocabStream: public Helper::Stream<string>{
  SrcStream& src;
  bool add_eos;
  std::queue<string> q;
  string eos_str{""};       // eos special mark

  void get_more_if_empty(){
    if(q.empty()){
      while(1){
        string line{};
        std::getline(src, line);
        if(!src)
          break;
        std::stringstream one_ss{line};
        string token;
        while(one_ss >> token)
          q.push(std::move(token));
        if(add_eos)
          q.push(eos_str);
        if(!q.empty())
          break;
      }
    }
  }
public:
  LineVocabStream(SrcStream& s, bool eos): src(s), add_eos(eos){}
  virtual Stream& operator>>(string& v){
    get_more_if_empty();
    utils::zcheck(!q.empty(), "The stream has been run up!");
    v = std::move(q.front());   // to be poped
    q.pop();
    return *this;
  }
  virtual operator bool(){
    get_more_if_empty();
    return !q.empty();
  }
};

// dictionary
class BasicVocab{
protected:
  typedef unordered_map<string, CountInt> VocabCountMap;
  typedef unordered_map<string, Index> VocabMap;
  typedef vector<string> VocabList;

  // read the vocab from file
  template<class SrcStream>
  static void build_vocab(SrcStream& input, const VocabList& pre_list, const VocabList& post_list,
      unique_ptr<VocabMap>& v, unique_ptr<VocabCountMap>& word_freqs){
    auto streamer = LineVocabStream<SrcStream>(input, false);
    word_freqs = Helper::count_freq<string, CountInt>(streamer);
    auto ranked_list = Helper::rank_key<string, CountInt>(*word_freqs);
    // write
    v = unique_ptr<VocabMap>(new VocabMap{});
    Index cur_id = 0;
    for(auto&& tok : pre_list)
      v->insert({tok, cur_id++});
    for(auto&& tok : *ranked_list)
      v->insert({tok, cur_id++});
    for(auto&& tok : post_list)
      v->insert({tok, cur_id++});
    return;
  }

  // called at init
  void construct_(){
    // get final_words
    final_words.resize(v.size());
    for(auto& p : v){
      auto idx = p.second;
      utils::zcheck(final_words[idx].size()==0, "Repeated idx!");
      final_words[idx] = p.first;
    }
  }

  // fields
  VocabMap v{};
  VocabCountMap word_freqs{};
  //VocabList pre_specials{};
  //VocabList post_specials{};
  VocabList final_words{};

  BasicVocab(){}

public:
  // constructor
  BasicVocab(std::istream& src_stream, const VocabList& pre_specials, const VocabList& post_specials){
    unique_ptr<VocabMap> v0; 
    unique_ptr<VocabCountMap> word_freqs0;
    build_vocab(src_stream, pre_specials, post_specials, v0, word_freqs0);
    v = std::move(*v0);
    word_freqs = std::move(*word_freqs0);
    construct_();
  }

  // saver & loader
  void write(std::ostream& output){
    for(Index i = 0; i < final_words.size(); i++)
      output << i << '\t' << final_words[i] << '\n';
    //for(auto& p : word_freqs)
    //  output << p.first << '\t' << p.second << '\n';
  }

  // get item
  const Index operator[](const string& one) const{
    return v.at(one);
  }

  const string& get_str(const Index i) const{
    return final_words.at(i);
  }

  // size
  const Index size() const{
    return v.size();
  }
};

#endif // !_H_LM_HELP
