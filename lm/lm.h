#ifndef _H_LM
#define _H_LM

#include "help.h"
#include <unordered_map>

namespace trie{
  //
  template<class T, class V>
  class Trie{
  public:
    struct TrieUnit{
    protected:
      typedef unordered_map<T, unique_ptr<TrieUnit>> Map;
      static Map default_map;
      unique_ptr<Map> map{nullptr};
    public:
      TrieUnit* find_or_insert(const T& tok, bool insert, bool* inserted = nullptr){
        TrieUnit* failure = nullptr;
        // first check map
        if(!this->map){
          if(insert){
            this->map.reset(new Map());
            if(inserted)
              *inserted = true;
          }
          else
            return failure;
        }
        // check the map
        auto finding = this->map->find(tok);
        if(finding == this->map->end()){
          if(insert){
            auto pp = std::make_pair(tok, std::move(unique_ptr<TrieUnit>{new TrieUnit()}));
            this->map->insert(std::move(pp));
            if(inserted)
              *inserted = true;
          }
          else
            return failure;
        }
        return map->at(tok).get();
      }
    public:
      V value{};
      decltype(map->begin()) begin(){ if(map) return map->begin(); else return default_map.begin(); }
      decltype(map->end()) end(){ if(map) return map->end(); else return default_map.end(); }
      const std::size_t size() const { return map->size(); }
    };

  protected:
    //
    TrieUnit root;   // the init point

    // methods
    // common for insert and query, bool-flag means whether to new one or return null
    template<class Iter>
    TrieUnit* find_or_insert(TrieUnit* start, Iter a, Iter b, bool insert, bool* inserted=nullptr){
      TrieUnit* cur = start;
      if(insert && inserted)
        *inserted = false;
      while(a!=b){
        cur = cur->find_or_insert(*a, insert, inserted);
        a = std::next(a);
        if(!cur)
          break;
      }
      return cur;
    }

    // recusively do DFS operations (modifiable on V)
    template<class Op>
    void dfs_recursive(const T& key, TrieUnit* cur, Op* oper=nullptr){
      if(oper)
        oper->push(key, cur);
      for(auto& ptr: *cur)
        dfs_recursive<Op>(ptr.first, ptr.second.get(), oper);
      if(oper)
        oper->pop();
    }

  public:
    template<class Op>
    void dfs(Op* oper){
      dfs_recursive<Op>(-1, &root, oper);
    }

  public:
    typedef TrieUnit Node;

    template<class Iter>
    void insert(Iter a, Iter b, const V& v){
      bool inserted = false;
      TrieUnit* one = find_or_insert(&root, a, b, true, &inserted);
      utils::zcheck(inserted, "Repeated adding of certain key!");
      one->value = v;
    }

    template<class Iter>
    V* query(Iter a, Iter b){
      TrieUnit* one = find_or_insert(&root, a, b, false);
      if(!one)
        return nullptr;
      else
        return &one->value;
    }

  };

  // static
  template<class T, class V>
  typename Trie<T, V>::TrieUnit::Map Trie<T, V>::TrieUnit::default_map;
};

using namespace trie;
// the final language model
template<class T>
class LM{
protected:
  typedef std::pair<CountInt, Real> Entry;
  Index ord;
  CountInt counts;
  Trie<T, Entry> data;    // phase1: (count, discount) --norm--> phase2: (count, log-prob/backoff)
  typedef typename Trie<T, Entry>::Node Node;

  // for norm
  class op_norm{
    Index ord;
    vector<Node*> stack;
  public:
    op_norm(Index order): ord(order){}
    void push(const T& key, Node* cur){
      stack.push_back(cur);
      Index cur_ord = stack.size();
      if(cur_ord == ord+1){
        // leaf node (plus the root NON node)
        auto prev = stack[stack.size() - 2];
        auto tmpv = Helper::log(Real(cur->value.first) - cur->value.second) - Helper::log(Real(prev->value.first));
        utils::zcheck(tmpv<=0, "Bad log-prob.");
        cur->value.second = tmpv;
      }
      else if(cur_ord == ord){
        // n-1 node, get backoff
        cur->value.second = Helper::log(cur->value.second) - Helper::log(Real(cur->value.first));
        utils::zcheck(cur->value.second <= 0, "Bad log-prob.");
      }
    }
    void pop(){
      stack.pop_back();
    }
  };

  // for interpolate
  class op_interpolate{
    Index ord;
    vector<Node*> stack;
    vector<T> seq;
    LM<T>* backto;
  public:
    op_interpolate(Index order, LM<T>* b): ord(order), backto(b){}
    void push(const T& key, Node* cur){
      stack.push_back(cur);
      seq.push_back(key);
      if(seq.size() == ord+1){
        // leaf
        Real backoff = (*(stack.end()-2))->value.second;
        Real backto_prob = backto->query(seq.begin()+2, seq.end())->second;   // should be there
        Entry& to_update = cur->value;
        to_update.second = Helper::logsumexp({to_update.second, backoff+backto_prob});
      }
    }
    void pop(){
      stack.pop_back();
      seq.pop_back();
    }
  };
  // special one for lowest order to push mass to DEFAULT order-0: average
  class op_interpolate_unk{
    Index ord;
    Index length{0};
    Index unk, bos;
    Real backoff_logp{0};
    static const int NON_PROB = -99;
    //
    void insert_extra(Node* cur, Index unk){
      bool inserted{false};
      Entry& to_update = cur->find_or_insert(unk, true, &inserted)->value;
      if(inserted)
        to_update.second = NON_PROB;
    }
  public:
    op_interpolate_unk(Index order, Index unk_idx, Index bos_idx): ord(order), unk(unk_idx), bos(bos_idx){}
    void push(const T& key, Node* cur){
      length++;
      if(length == ord){
        // step1: get backoff value
        insert_extra(cur, unk);
        backoff_logp = cur->value.second - Helper::log(cur->size());
        insert_extra(cur, bos);   // bos will not receive
      }
      else if(length == ord + 1){
        // step2: add to others
        if(key != bos){
          Entry& to_update = cur->value;
          to_update.second = Helper::logsumexp({to_update.second, backoff_logp});
        }
      }
    }
    void pop(){
      length--;
    }
  };

  class op_write_arpa{
    Index ord;
    std::ostream& output;
    const BasicVocab& vocab;
    LM<T>* higher;
    vector<T> seq;
  public:
    op_write_arpa(Index ord_, std::ostream& output_, const BasicVocab& vocab_, LM<T>* higher_):
      ord(ord_), output(output_), vocab(vocab_), higher(higher_){}
    void push(const T& key, Node* cur){
      seq.push_back(key);
      if(seq.size() == ord+1){
        // leaf
        output << cur->value.second << '\t';
        for(auto i=seq.begin()+1;i!=seq.end();i++)
          output << vocab.get_str(*i) << ((i!=seq.end()-1) ? ' ' : '\t');
        if(higher){
          Entry* higher_bnode = higher->query(seq.begin()+1, seq.end(), -1);
          if(higher_bnode)
            output << higher_bnode->second;
        }
        output << '\n';
      }
    }
    void pop(){
      seq.pop_back();
    }
  };

public:
  LM(Index order): ord(order), counts(0){}
  const CountInt size() const { return counts; }
  // phase1: the process of building
  template<class Iter>
  void add(Iter a, Iter b, const CountInt count, const Real discount){
    utils::zcheck(std::distance(a, b) == ord, "Unmatched order");
    utils::zcheck(Real(count) > discount, "Unreasonable discount");
    // insert into the data
    data.insert(a, b, std::make_pair(count, discount));
    counts++;
    auto* ctx_v = data.query(a, b-1);   // TODO: could be more efficient
    ctx_v->first += count;
    ctx_v->second += discount;
  }
  // phase2: normalize all, (TODO): norm at each points could be more efficient
  void norm_all(){
    op_norm z(ord);
    data.dfs(&z);
  }
  // query for value
  template<class Iter>
  Entry* query(Iter a, Iter b, int bias=0){
    utils::zcheck(std::distance(a, b)-bias == ord, "Unmatched order");
    return data.query(a, b);
  }
  // interpolate
  void interpolate(LM<T>* backend){
    utils::zcheck(ord == backend->ord + 1, "Bad interpolate backoff order.");
    op_interpolate z(ord, backend);
    data.dfs(&z);
  }
  // no interpolate, but push the mass to unk (for lowest order)
  void interpolate(const T& UNK, const T& BOS){
    op_interpolate_unk z(ord, UNK, BOS);
    data.dfs(&z);
  }
  // write
  void write_arpa(std::ostream& output, const BasicVocab& vocab, LM<T>* higher){
    output << "\\" << ord << "-grams:\n";
    op_write_arpa z(ord, output, vocab, higher);
    data.dfs(&z);
    output << "\n";
  }
};

template<class T>
class LMGroup{
protected:
  Index highest_ord;
  vector<LM<T>> lms;    // 0: highest, ..., -1: 1-gram (todo: could be more compact)
public:
  LMGroup(Index ord): highest_ord(ord){
    utils::zcheck(highest_ord > 0, "Unlegal n-gram order.");
    for(Index i = 0; i < ord; i++)
      lms.emplace_back(LM<T>{ord - i});
  }
  // building progress
  template<class Iter>
  void add(Iter a, Iter b, const CountInt count, const Real discount){
    auto cur_ord = std::distance(a, b);
    utils::zcheck(cur_ord>0 && Index(cur_ord)<=highest_ord, "Unlegal order");
    lms[highest_ord-cur_ord].add(a, b, count, discount);
  }
  void norm_all(){
    for(auto& m : lms)
      m.norm_all();
  }
  // from lower order to higher one
  void interpolate(const T& UNK, const T& BOS){
    lms[highest_ord-1].interpolate(UNK, BOS);
    for(Index i=highest_ord; i>=2; i--)
      lms[i-2].interpolate(&lms[i-1]);
  }
  // final output
  void write_arpa(std::ostream& output, const BasicVocab& vocab){
    // head
    output << "\\data\\\n";
    for(Index i = 1; i <= highest_ord; i++)
      output << "ngram " << i << "=" << lms[highest_ord-i].size() << "\n";
    output << "\n";
    // n-grams
    for(Index i = highest_ord; i >= 1; i--){
      LM<T>* cur = &lms[i-1];
      LM<T>* higher = nullptr;
      if(i>=2)
        higher = &lms[i-2];
      cur->write_arpa(output, vocab, higher);
    }
    output << "\\end\\";
  }
};

#endif
