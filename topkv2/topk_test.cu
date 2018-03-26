#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <unordered_set>
#include <vector>
#include <cstdio>
#include <typeinfo>
#include <string>

#define HAVE_CUDA 1
#include "topk.h"
#include <Eigen/Eigen>

// the testing
void CUDA_CHECK(cudaError_t ret){
  if(ret != cudaSuccess) {
    std::cerr << "CUDA failure in " << cudaGetErrorString(ret) << std::endl;
    throw std::runtime_error("");
  }
}

Eigen::GpuDevice* edevice;
void setup(int gpuid){
  auto estream = new Eigen::CudaStreamDevice(gpuid);
  edevice = new Eigen::GpuDevice(estream);
  CUDA_CHECK(cudaSetDevice(gpuid));
  srand(12345);
}

using std::unordered_set;
using std::vector;
using std::cout;
using std::endl;
using std::clock;

double get_rand(int bits){
  double x = 0.;
  for(int i = 0; i < bits; i++){
    x += double(rand()) / RAND_MAX;
    x *= 10;
  }
  if(double(rand()) / RAND_MAX > 0.5)
    x *= -1;
  return x;
}

//#define DEBUG
template<typename T, typename IndexType>
void check(IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
  vector<unordered_set<IndexType>>& answer, T* v_input, T* v_val, IndexType* v_idx){
  int bad_count = 0;
  for(int i = 0; i < outer_size*inner_size; i++){
#ifdef DEBUG
    auto cur_input = input + i/inner_size*inner_size*cur_size + i%inner_size;
    auto cur_val = v_val + i/inner_size*inner_size*k + i%inner_size;
#endif // DEBUG
    auto cur_idx = v_idx + i/inner_size*inner_size*k + i%inner_size;
    const unordered_set<IndexType>& gold = answer[i];
    unordered_set<IndexType> pred;
    for(int j = 0; j < k; j++)
      pred.insert(cur_idx[j*inner_size]);
    if(gold != pred){
      bad_count++;
#ifdef DEBUG
      // sort the results
      using PairType = std::pair<IndexType, T>;
      vector<PairType> gold_ranks, pred_ranks;
      for(auto idx : gold)
        gold_ranks.push_back(std::make_pair(idx, cur_input[idx*inner_size]));
      for(int j = 0; j < K; j++)
        pred_ranks.push_back(std::make_pair(cur_idx[j*inner_size], cur_val[j*inner_size]));
      std::sort(gold_ranks.begin(), gold_ranks.end(), [](const PairType& a, const PairType& b){ return a.second> b.second; });
      std::sort(pred_ranks.begin(), pred_ranks.end(), [](const PairType& a, const PairType& b){ return a.second> b.second; });
      //
      cout << "Check unequal of batch-id " << i << ", the results are:" << endl;
      for(int j = 0; j < K; j++)
        cout << gold_ranks[j].first << "=" << gold_ranks[j].second << "\t";
      cout << endl;
      for(int j = 0; j < K; j++)
        cout << pred_ranks[j].first << "=" << pred_ranks[j].second << "\t";
      cout << endl;
      std::cin.get();
#endif // DEBUG
    }
  }
#ifndef DEBUG
  if(bad_count > 0)
    cout << "Check unequal of numbers of " << bad_count << endl;
#endif
}

// checking type
template<typename T, typename U>
struct is_same
{
  static constexpr bool value = false;
};

template<typename T>
struct is_same<T, T>
{
  static constexpr bool value = true;
};

template<typename T, typename U>
constexpr bool eqTypes() { return is_same<T, U>::value; }

#define NOPE_CODE -1000
#define LOCAL_REPORT(tag0, tag1)                  \
    timek /= CLOCKS_PER_SEC;                      \
    timec /= CLOCKS_PER_SEC;                      \
    std::cout << tag0;                            \
    if(tag1 != NOPE_CODE)                         \
      cout << "-" << tag1;                        \
    std::cout << "\t" << typeid(T).name() << '\t' \
      << typeid(IndexType).name() << '\t'         \
      << outer_size << "\t"                       \
      << inner_size << "\t" << cur_size << "\t"   \
      << k << "\t" << steps << "\t" << timec << "\t"      \
      << timek << "\t" << timec/steps << "\t"     \
      << timek/steps << std::endl;                \

// testing
template<typename T, typename IndexType>
void test(IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k, IndexType steps){
  // allocating
  size_t size_input = outer_size * inner_size * cur_size * sizeof(T);
  size_t size_val = outer_size * inner_size * k * sizeof(T);
  size_t size_idx = outer_size * inner_size * k * sizeof(IndexType);
  auto v_input = new T[outer_size * inner_size * cur_size];
  auto v_val = new T[outer_size * inner_size * k];
  auto v_idx = new IndexType[outer_size * inner_size * k];
#ifdef DEBUG
  for(int i = 0; i < outer_size * inner_size * cur_size; i++)
    v_input[i] = T(i%(inner_size * cur_size));
#else
  for(int i = 0; i < outer_size * inner_size * cur_size; i++)
    v_input[i] = T(get_rand(10));
#endif
  //
  cout << "--- Start a test: T=" << typeid(T).name() << " IndexType=" << typeid(IndexType).name() << ' '
    << outer_size << ' ' << inner_size << ' ' << cur_size << ' ' << k << ' ' << steps << endl;
  // sort on cpu as the gold answer
  //
  vector<unordered_set<IndexType>> answers(outer_size*inner_size);
  {
    cout << "== Test with cpu-sorting" << endl;
    auto t1 = clock();
    using PairType = std::pair<T, IndexType>;
    for(int i = 0; i < outer_size*inner_size; i++){
      auto cur_input = v_input + i/inner_size*inner_size*cur_size + i%inner_size;
      vector<PairType> result;
      for(int j = 0; j < cur_size; j++)
        result.push_back(std::make_pair(cur_input[j*inner_size], j));
      std::sort(result.begin(), result.end(), [](const PairType& a, const PairType& b){ return a.first>b.first; });
      for(int j = 0; j < k; j++)
        answers[i].insert(result[j].second);
    }
    auto t2 = clock();
    //
    double time_once = double(t2 - t1);
    double timek = time_once*steps;
    double timec = 0;
    LOCAL_REPORT("sort", NOPE_CODE);
  }
  //
  {
    cout << "== Test with topk-cpu" << endl;
    double timek = 0;
    double timec = 0;
    for(int i = 0; i < steps; i++){
      auto t1 = clock();
      topk_cpu::topk<T, IndexType, true>(v_input, v_val, v_idx, outer_size, inner_size, cur_size, k);
      auto t2 = clock();
      timek += (t2 - t1);
    }
    //
    check(outer_size, inner_size, cur_size, k, answers, v_input, v_val, v_idx);
    LOCAL_REPORT("topk-cpu", NOPE_CODE);
  }
  // alloc and copy
  // Allocate memory for each vector on GPU
  T *cv_input;
  T *cv_val;
  IndexType *cv_idx;
  cudaMalloc(&cv_input, size_input);
  cudaMalloc(&cv_val, size_val);
  cudaMalloc(&cv_idx, size_idx);
  cudaMemcpy(cv_input, v_input, size_input, cudaMemcpyHostToDevice);
  //
  try{
    cout << "== Test with topk-gpu" << endl;
    double timek = 0;
    double timec = 0;
    // calculate
    for(int i = 0; i < steps; i++){
      auto t1 = clock();
      CUDA_CHECK(topk_gpu::topk<T, IndexType, true, topk_gpu::TopK_Gpu_Strategy::TOPK_AUTO>(cv_input, cv_val, cv_idx, outer_size, inner_size, cur_size, k));
      cudaDeviceSynchronize();
      auto t2 = clock();
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
      auto t3 = clock();
      timek += (t2 - t1);
      timec += (t3 - t2);
    }
    check(outer_size, inner_size, cur_size, k, answers, v_input, v_val, v_idx);
    LOCAL_REPORT("topk-gpu(auto)", NOPE_CODE);
  }
  catch(...){ cout << "error" << endl; }
  //
  try{
    cout << "== Test with impl_eigen" << endl;
    double timek = 0;
    double timec = 0;
    // calculate
    for(int i = 0; i < steps; i++){
      auto t1 = clock();
      CUDA_CHECK(impl_eigen::topk(cv_input, cv_val, cv_idx, outer_size, inner_size, cur_size, k, *edevice));
      cudaDeviceSynchronize();
      auto t2 = clock();
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
      auto t3 = clock();
      timek += (t2 - t1);
      timec += (t3 - t2);
    }
    check(outer_size, inner_size, cur_size, k, answers, v_input, v_val, v_idx);
    LOCAL_REPORT("topk-eigen", NOPE_CODE);
  }
  catch(...){ cout << "error" << endl; }
  //
  try{
    IndexType max_shards = impl_tf::get_max_num_shards<T, IndexType>(k, cur_size);
    IndexType auto_shards = impl_tf::get_auto_num_shards<T, IndexType>(k, cur_size);
    cout << "== Test with impl_tf, max-shard is " << max_shards << endl;
    for(IndexType num_shards : {(int)auto_shards, 8, 16, 32, 64, 96, 128, 256}){
      if(num_shards>=max_shards)
        continue;
      double timek = 0;
      double timec = 0;
      // calculate
      for(int i = 0; i < steps; i++){
        auto t1 = clock();
        CUDA_CHECK(impl_tf::topk<T, IndexType, true>(cv_input, cv_val, cv_idx, outer_size, inner_size, cur_size, k, num_shards));
        cudaDeviceSynchronize();
        auto t2 = clock();
        // copy back
        cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
        auto t3 = clock();
        timek += (t2 - t1);
        timec += (t3 - t2);
      }
      check(outer_size, inner_size, cur_size, k, answers, v_input, v_val, v_idx);
      LOCAL_REPORT("topk-tf", num_shards);
    }
  }
  catch(...){ cout << "error" << endl; }
  //
  try{
    cout << "== Test with impl_tr" << endl;
    double timek = 0;
    double timec = 0;
    // calculate
    for(int i = 0; i < steps; i++){
      auto t1 = clock();
      CUDA_CHECK(impl_tr::topk<T, IndexType, true>(cv_input, cv_val, cv_idx, outer_size, inner_size, cur_size, k));
      cudaDeviceSynchronize();
      auto t2 = clock();
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
      auto t3 = clock();
      timek += (t2 - t1);
      timec += (t3 - t2);
    }
    check(outer_size, inner_size, cur_size, k, answers, v_input, v_val, v_idx);
    LOCAL_REPORT("topk-tr", NOPE_CODE);
  }
  catch(...){ cout << "error" << endl; }
  // free
  cudaFree(cv_input);
  cudaFree(cv_val);
  cudaFree(cv_idx);
  cout << endl;
  return;
}

int main_topk()
{

#define LOCAL_TEST()                                                             \
  int steps = 100;                                                               \
  test<float, int>(outer_size, inner_size, cur_size, k, steps);                  \
  test<double, int>(outer_size, inner_size, cur_size, k, steps);                 \
  test<float, Eigen::DenseIndex>(outer_size, inner_size, cur_size, k, steps);    \
  test<double, Eigen::DenseIndex>(outer_size, inner_size, cur_size, k, steps);   \

  setup(2);
  /*
  // run specific sets
  // r1
  for(int outer_size: {1, 32, 80,}){   // batch-size
    for(int inner_size: {1, 8, 32}){
      for(int cur_size: {100, 1000, 10000, 100000}){
        for(int k: {1, 5, 10, 15, 20, 25, 30, 50, 80, 100}){
          LOCAL_TEST();
        }
      }
    }
  }
  // r2
  for(int outer_size : {1, 32, 80, 128}){   // batch-size
    for(int inner_size : {1, 4, 8, 32, 64, 80}){
      for(int cur_size : {1000, 10000,}){
        for(int k : {1, 10, 20}){
          LOCAL_TEST();
        }
      }
    }
  }
  */
  /*// r3 on x48
  for(int outer_size : {16, 80,}){   // batch-size
    for(int inner_size : {1, 4, 8}){
      for(int cur_size : {1000, 30000, }){
        for(int k : {1, 5, 10, 16, 20, 32, 50, }){
          LOCAL_TEST();
        }
      }
    }
  }*/
  for(int outer_size : {80, }){   // batch-size
    for(int inner_size : {1, 4, }){
      for(int cur_size : {1000, 30000, }){
        for(int k : {1, 4, 16, 32, 64, }){
          LOCAL_TEST();
        }
      }
    }
  }
  return 0;
}

// mv topk_test.cpp topk_test.cu
// compile: nvcc -std=c++11 -g -DEIGEN_USE_GPU -I../../libs/eigen-eigen-5a0156e40feb-334/ ./topk_test.cu -o run
// compile: nvcc -std=c++11 -O3 -DEIGEN_USE_GPU -I../../libs/eigen-eigen-5a0156e40feb-334/ ./topk_test.cu -o debug
// ./run | grep -v -E "^-|^=|^$|error"

#define RUN_TOPK
#ifdef RUN_TOPK
int main(){
  main_topk();
}
#endif // RUN_TOPK

