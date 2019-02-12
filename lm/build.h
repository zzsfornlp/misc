#ifndef _H_NGRAM
#define _H_NGRAM

#include "help.h"
#include "lm.h"

// representing one ngram
template<class T>
class Ngram{
private:
  vector<T> ngram;
  vector<CountInt> counts;    // -1: 1-gram, -2: 2-gram, ... or 0:n-gram, 1:(n-1)-gram, ...
  void resize(){ counts.resize(ngram.size(), 0); }
public:
  Ngram(const vector<T>& window): ngram(window){}
  Ngram(const Ngram<T>& n): ngram(n.ngram), counts(n.counts){}
  const std::size_t size() const { return ngram.size(); }
  const T& operator[](Index idx) const { return ngram[idx]; }
  const vector<T>& get_ngram() const{ return ngram; }
  // get counts
  CountInt get_count(Index which){
    if(which <= counts.size())
      resize();
    return counts[which];
  }
  // operations used when real counting (zero for query)
  CountInt adjust_count(Index which, CountInt delta = 0){
    if(which <= counts.size())
      resize();
    counts[which] += delta;
    return counts[which];
  }
  // Direct setting
  void set_count(Index which, CountInt c){
    if(which <= counts.size())
      resize();
    counts[which] = c;
  }
};

// representing Ngram and Ngram-set
template<class T>
class NgramSlider{
private:
  Index order;
  T default_val;
  vector<T> window;
public:
  NgramSlider(Index ord, const T& default_v): order(ord), default_val(default_v), window(ord, default_v){}

  Ngram<T> add(const T& v){
    // move one step to the left
    for(Index i = 0; i + 1<order; i++)
      window[i] = window[i + 1];
    window.back() = v;
    return Ngram<T>(window);
  }

  void clear(){
    std::fill(window.begin(), window.end(), default_val);
  }
};

// count of count statistics
class CoC{
  static const CountInt N = 3;
private:
  vector<CountInt> counts;  // 0, 1, 2, 3, 4
  vector<Real> discounts;   // 0:0, N1, N2, N3+
public:
  CoC(): counts(N+2, 0){}
  void add(CountInt x){
    utils::zcheck(x > 0, "why zero here?");
    if(x < (N+2))
      counts[x] += 1;
  }
  // final calculate
  void calc(const vector<Real>& fallback){
    discounts.resize(N+1, 0);
    Real Y = Real(counts[1])/(counts[1]+2*counts[2]);
    for(CountInt i = 1; i <= N; i++){
      if(counts[i + 1] == 0 || counts[i] == 0){
        utils::zlog(utils::string_format("Warn: Fallback on count==%d to %f.", i, fallback[i]));
        discounts[i] = fallback[i];
      }
      else
        discounts[i] = i - (i + 1)*Y*counts[i + 1] / counts[i];
      //
      if(discounts[i] < 0){
        utils::zlog(utils::string_format("Warn: %f<=0, thus fallback on count==%d to %f.", discounts[i], i, fallback[i]));
        discounts[i] = fallback[i];
      }
    }
  }
  Real get_d(CountInt x) const{
    if(x>N) x=N;
    return discounts[x];
  }
};

// simple in-memory data representation, could be extended
template<class T>
class NgramCollection{
private:
  vector<Ngram<T>> data;
  Index ord;
  // statistics
  vector<CoC> cocs;   // 0: ord-gram, ..., -1: 1-gram
public:
  const decltype(ord) get_ord() const { return ord; }
  const CoC& get_coc(Index i) const { return cocs[i]; }
  decltype(data.begin()) begin(){ return data.begin(); }
  decltype(data.end()) end(){ return data.end(); }
  // construct incrementally
  void add(Ngram<T>&& one){
    data.emplace_back(std::move(one));
    auto this_size = data.back().size();
    if(data.size() == 1)
      ord = this_size;
    else
      utils::zcheck(ord == this_size, "Unmatched length of newly added ngram!");
  }
  //
  Ngram<T>& back(){ return data.back(); }
  void assign_coc(vector<CoC>&& v){ cocs = std::move(v); }
  const CoC& get_coc(const Index i){ return cocs[i]; }
};

// the default in-memory sorter
enum class CompOrder{ SuffixOrder, ContextOrder, PrefixOrder };
template<CompOrder comp_order>
class Sorter{
public:
  // sort related
  static vector<Index> get_comp_order(const CountInt n){
    vector<Index> ret(n);
    for(Index i = 0; i < n; i++) // prefix as init
      ret[i] = i;
    switch(comp_order)
    {
    case CompOrder::SuffixOrder: std::reverse(ret.begin(), ret.end()); break;
    case CompOrder::ContextOrder: if(n >(CountInt)1){ std::reverse(ret.begin(), ret.end() - 1); }; break;
    case CompOrder::PrefixOrder: break;
    default: throw comp_order; break;
    }
    return ret;
  }
  template<class CC>
  struct Comp{
    const vector<Index>& comp_order_;
    Comp(const vector<Index>& c): comp_order_(c){}
    inline bool operator() (const CC& lhs, const CC& rhs) {
      for(auto i : comp_order_) {
        if(lhs[i] != rhs[i])
          return lhs[i] < rhs[i];
      }
      return false;
    }
  };
  template<class SS>
  static void sort_inplace(SS& data){
    auto order_vec = get_comp_order(data.get_ord());
    std::sort(data.begin(), data.end(), Comp<decltype(*data.begin())>(order_vec));
  }
};

// adjusted counts for modified-KN & get the interpolated probabilities
template<class T>
class LMBuilder{
private:
  // the first index of difference
  template<CompOrder EnumVal>
  static SignedIndex find_difference(const Ngram<T>& a, const Ngram<T>& b){
    utils::zcheck(a.size()==b.size(), "The two n-grams has different n.");
    vector<Index> comp_order = Sorter<EnumVal>::get_comp_order(a.size());
    for(Index i : comp_order)
      if(a[i] != b[i])
        return i;
    return -1;
  }

public:
  // copy into another NgramCollection (TODO: can be inplaced), require SUFFIX-SORTED input
  // - this scanning of counting is up from higher-order to lower-order, with special dealing of BOS
  static unique_ptr<NgramCollection<Index>> count_on_ngrams(NgramCollection<T>& data, const T& BOS, const vector<Real>& coc_back){
    auto col_counts = unique_ptr<NgramCollection<Index>>(new NgramCollection<Index>());
    auto order = data.get_ord();
    vector<CoC> cocs(order);    // count of counts stat
    // sort on SuffixOrder
    Sorter<CompOrder::SuffixOrder>::sort_inplace(data);
    // scan
    Ngram<T>* prev{new Ngram<T>(vector<T>(order, BOS))};   // the first padding one or back of collection
    unique_ptr<Ngram<T>> prev_release_ptr{prev};        // to be released
    for(Ngram<T>& one : data){
      // first index of difference
      SignedIndex diff_index = find_difference<CompOrder::SuffixOrder>(one, *prev);
      if(diff_index < 0){
        // equal ngram, ignore the later one and simple add to prev
        prev->adjust_count(0, 1);   // highest-gram += 1
        for(Index i = 1; i < order; i++){
          // add only when hitting special case of BOS, otherwise ignore for KN
          if(one[i] == BOS)
            prev->adjust_count(i, 1);
          else
            break;
        }
      }
      else{
        Ngram<T> cur{one};
        Index diff_index_uns = diff_index;  // >0
        for(Index i = 0; i < order; i++){
          if(i <= diff_index_uns){  // different k-gram, summarize the previous one
            cur.adjust_count(i, 1);
            if(prev != prev_release_ptr.get())  // not counting the dummy starting point
              cocs[i].add(prev->get_count(i));
          }
          else if(i == diff_index_uns + 1){   // same k-gram, but different prefix, +1
            cur.adjust_count(i, prev->get_count(i)+1);
            prev->set_count(i, 0);  // moved to the new one
          }
          else{   // +1 only for BOS
            cur.adjust_count(i, prev->get_count(i));
            prev->set_count(i, 0);  // moved to the new one
            if(cur[i] == BOS)
              cur.adjust_count(i, 1);
          }
        }
        // add and replace prev
        col_counts->add(std::move(cur));
        prev = &col_counts->back();
      }
    }
    // final for CoC
    for(Index i = 0; i < order; i++){
      cocs[i].add(prev->get_count(i));
      cocs[i].calc(coc_back);
    }
    col_counts->assign_coc(std::move(cocs));
    return col_counts;
  }

  // input should have counts and coc, and also be sorted by context
  static unique_ptr<LMGroup<T>> build_lm(NgramCollection<T>& counts){
    Index order = counts.get_ord();
    auto lm = unique_ptr<LMGroup<T>>(new LMGroup<T>(order));
    /* // todo: could be more effective by utilizing the ContextOrder
    // sort to be more efficient: todo
    Sorter<CompOrder::ContextOrder>::sort_inplace(counts);
    // scan
    Ngram<T>* prev{new Ngram<T>(vector(order, BOS))};   // the first padding one or back of collection
    unique_ptr<Ngram<T>> prev_release_ptr{prev};        // to be released
    for(Ngram<T>& one : counts){
      // first index of difference
      SignedIndex diff_index = find_difference<CompOrder::ContextOrder>(one, *prev);
      utils::zcheck(diff_index > 0, "Sth wrong in previous counting procedure!");
      if(diff_index == SignedIndex(order) - 1){
        // only the final un-contexted token differs
      }
      else{
        // also the ending of certain contexts
      }
    }
    */
    // init prob
    for(Ngram<T>& one : counts){
      auto ng = one.get_ngram();
      Index idx = 0;
      for(auto iter = ng.begin(); iter != ng.end(); iter++, idx++){
        CountInt cc = one.get_count(idx);
        Real disc = counts.get_coc(idx).get_d(cc);
        if(cc>0)
          lm->add(iter, ng.end(), cc, disc);
      }
    }
    lm->norm_all();
    return lm;
  }
};

#endif // !_H_NGRAM
