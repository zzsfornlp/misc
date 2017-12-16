#include <iostream>
#include <cstdlib>
#include <stdexcept>
#include <ctime>
#include <assert.h>
#include <algorithm>
#include <unordered_set>
#include <vector>
#include <cstdio>
#include <typeinfo>

#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

/*
#define __device__
#define __shared__
#define __restrict__
#define __global__
#define __forceinline__
*/

// pytroch's implementation using radix-selection
// - from pytorch/aten/src/THC/THCTensorTopK.cuh
// - warning: ignore some types, mainly for float; especially does not include THCNumerics for T==half
namespace impl_tr{
  // 
  template <typename T>
  struct TopKTypeConfig {};

  template <>
  struct TopKTypeConfig<float> {
    typedef uint32_t RadixType;
    // Converts a float to an integer representation with the same
    // sorting; i.e., for floats f1, f2:
    // if f1 < f2 then convert(f1) < convert(f2)
    // We use this to enable radix selection of floating-point values.
    // This also gives a relative order for NaNs, but that's ok, as they
    // will all be adjacent
    static inline __device__ RadixType convert(float v) {
      RadixType x = __float_as_int(v);
      RadixType mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
      return (x ^ mask);
    }
    static inline __device__ float deconvert(RadixType v) {
      RadixType mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
      return __int_as_float(v ^ mask);
    }
  };

  template <>
  struct TopKTypeConfig<double> {
    typedef uint64_t RadixType;

    static inline __device__ RadixType convert(double v) {
      RadixType x = __double_as_longlong(v);
      RadixType mask = -((x >> 63)) | 0x8000000000000000;
      return (x ^ mask);
    }

    static inline __device__ double deconvert(RadixType v) {
      RadixType mask = ((v >> 63) - 1) | 0x8000000000000000;
      return __longlong_as_double(v ^ mask);
    }
  };

  // helpers

  // For CC 3.5+, perform a load using __ldg
  template <typename T>
  __device__ __forceinline__ T doLdg(const T* p) {
#if __CUDA_ARCH__ >= 350
    return __ldg(p);
#else
    return *p;
#endif
  }

  // Collection of direct PTX functions
  template <typename T>
  struct Bitfield {};
  template <>
  struct Bitfield<unsigned int> {
    static __device__ __forceinline__
      unsigned int getBitfield(unsigned int val, int pos, int len) {
      unsigned int ret;
      asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(val), "r"(pos), "r"(len));
      return ret;
    }
    static __device__ __forceinline__
      unsigned int setBitfield(unsigned int val, unsigned int toInsert, int pos, int len) {
      unsigned int ret;
      asm("bfi.b32 %0, %1, %2, %3, %4;" :
      "=r"(ret) : "r"(toInsert), "r"(val), "r"(pos), "r"(len));
      return ret;
    }
  };

  template <>
  struct Bitfield<uint64_t> {
    static __device__ __forceinline__
      uint64_t getBitfield(uint64_t val, int pos, int len) {
      uint64_t ret;
      asm("bfe.u64 %0, %1, %2, %3;" : "=l"(ret) : "l"(val), "r"(pos), "r"(len));
      return ret;
  }

    static __device__ __forceinline__
      uint64_t setBitfield(uint64_t val, uint64_t toInsert, int pos, int len) {
      uint64_t ret;
      asm("bfi.b64 %0, %1, %2, %3, %4;" :
      "=l"(ret) : "l"(toInsert), "l"(val), "r"(pos), "r"(len));
      return ret;
    }
};

  // WARP_BALLOT
  __device__ __forceinline__ int WARP_BALLOT(int predicate, unsigned int mask = 0xffffffff)
  {
#if CUDA_VERSION >= 9000
    return __ballot_sync(mask, predicate);
#else
    return __ballot(predicate);
#endif
  }

  // ACTIVE_MASK
  __device__ __forceinline__ unsigned int ACTIVE_MASK()
  {
#if CUDA_VERSION >= 9000
    return __activemask();
#else
    // will be ignored anyway
    return 0xffffffff;
#endif
  }

  // getLaneId
  __device__ __forceinline__ int getLaneId() {
    int laneId;
    asm("mov.s32 %0, %laneid;" : "=r"(laneId));
    return laneId;
  }

  /**
  Computes ceil(a / b)
  */
  template <typename T>
  __host__ __device__ __forceinline__ T THCCeilDiv(T a, T b) {
    return (a + b - 1) / b;
  }

  /**
  Computes ceil(a / b) * b; i.e., rounds up `a` to the next highest
  multiple of b
  */
  template <typename T>
  __host__ __device__ __forceinline__ T THCRoundUp(T a, T b) {
    return THCCeilDiv(a, b) * b;
  }


  // This function counts the distribution of all input values in a
  // slice we are selecting by radix digit at `radixDigitPos`, but only
  // those that pass the filter `((v & desiredMask) == desired)`.
  // This produces and broadcasts the seen counts for a single block only.
  // `smem` must have at least `RadixSize` elements.
  template <typename DataType, typename BitDataType,
    typename IndexType, typename CountType,
    int RadixSize, int RadixBits>
    __device__ void countRadixUsingMask(CountType counts[RadixSize],
      CountType* smem,
      BitDataType desired,
      BitDataType desiredMask,
      int radixDigitPos,
      IndexType sliceSize,
      IndexType withinSliceStride,
      DataType* data) {
    // Clear out per-thread counts from a previous round
#pragma unroll
    for(int i = 0; i < RadixSize; ++i) {
      counts[i] = 0;
    }

    if(threadIdx.x < RadixSize) {
      smem[threadIdx.x] = 0;
    }
    __syncthreads();

    // Scan over all the data. Upon a read, the warp will accumulate
    // counts per each digit in the radix using warp voting.
    for(IndexType i = threadIdx.x; i < sliceSize; i += blockDim.x) {
      BitDataType val = TopKTypeConfig<DataType>::convert(doLdg(&data[i * withinSliceStride]));

      bool hasVal = ((val & desiredMask) == desired);
      BitDataType digitInRadix = Bitfield<BitDataType>::getBitfield(val, radixDigitPos, RadixBits);

#pragma unroll
      for(unsigned int j = 0; j < RadixSize; ++j) {
        bool vote = hasVal && (digitInRadix == j);
        counts[j] += __popc(WARP_BALLOT(vote, ACTIVE_MASK()));
      }
    }

    // Now, for each warp, sum values
    if(getLaneId() == 0) {
#pragma unroll
      for(unsigned int i = 0; i < RadixSize; ++i) {
        atomicAdd(&smem[i], counts[i]);
      }
    }

    __syncthreads();

    // For each thread, read in the total counts
#pragma unroll
    for(unsigned int i = 0; i < RadixSize; ++i) {
      counts[i] = smem[i];
    }

    __syncthreads();
  }

  // Over what radix we are selecting values
#define RADIX_BITS 2 // digits are base-(2 ^ RADIX_BITS)
#define RADIX_SIZE 4 // 2 ^ RADIX_BITS
#define RADIX_MASK (RADIX_SIZE - 1)

  // 
  // This finds the unique value `v` that matches the pattern
  // ((v & desired) == desiredMask) in our sorted int format
  template <typename DataType, typename BitDataType, typename IndexType>
  __device__ DataType findPattern(DataType* smem,
    DataType* data,
    IndexType sliceSize,
    IndexType withinSliceStride,
    BitDataType desired,
    BitDataType desiredMask) {
    if(threadIdx.x < 32) {
      smem[threadIdx.x] = (DataType)(0);
    }
    __syncthreads();

    // All threads participate in the loop, in order to sync on the flag
    IndexType numIterations = THCRoundUp(sliceSize, (IndexType)blockDim.x);
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < sliceSize);
      DataType v = inRange ? doLdg(&data[i * withinSliceStride]) : (DataType)(0);

      if(inRange && ((TopKTypeConfig<DataType>::convert(v) & desiredMask) == desired)) {
        // There should not be conflicts if we are using findPattern,
        // since the result is unique
        smem[0] = (DataType)(1);
        smem[1] = v; // can't use val as the flag, since it could be 0
      }

      __syncthreads();

      DataType found = smem[0];
      DataType val = smem[1];

      __syncthreads();

      // Check to see if a thread found the value
      if(found != (DataType)(0)) {
        // all threads return this value
        return val;
      }
    }

    // should not get here
    assert(false);
    return (DataType)(0);
  }

  // Returns the top-Kth element found in the data using radix selection
  // - find k out of sliceSize with stride of withinSliceStride, Descending if Order
  template <typename DataType, typename BitDataType, typename IndexType, bool Order>
  __device__ void radixSelect(DataType* data,
    IndexType k,
    IndexType sliceSize,
    IndexType withinSliceStride,
    int* smem,
    DataType* topK) {
    // Per-thread buckets into which we accumulate digit counts in our
    // radix
    int counts[RADIX_SIZE];
    // We only consider elements x such that (x & desiredMask) == desired
    // Initially, we consider all elements of the array, so the above
    // statement is true regardless of input.
    BitDataType desired = 0;
    BitDataType desiredMask = 0;
    // We are looking for the top kToFind-th element when iterating over
    // digits; this count gets reduced by elimination when counting
    // successive digits
    int kToFind = k;
    // We start at the most significant digit in our radix, scanning
    // through to the least significant digit
#pragma unroll
    for(int digitPos = sizeof(DataType) * 8 - RADIX_BITS;
      digitPos >= 0;
      digitPos -= RADIX_BITS) {
      // Count radix distribution for the current position and reduce
      // across all threads
      countRadixUsingMask<DataType, BitDataType,
        IndexType, int,
        RADIX_SIZE, RADIX_BITS>(
          counts, smem,
          desired, desiredMask, digitPos,
          sliceSize, withinSliceStride, data);
      // All threads participate in the comparisons below to know the
      // final result
#define CHECK_RADIX(i)                                                  \
    int count = counts[i];                                              \
                                                                        \
    /* All threads have the same value in counts here, so all */        \
    /* threads will return from the function. */                        \
    if (count == 1 && kToFind == 1) {                                   \
      /* There is a unique answer. */                                   \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The answer is now the unique element v such that: */           \
      /* (v & desiredMask) == desired */                                \
      /* However, we do not yet know what the actual element is. We */  \
      /* need to perform a search through the data to find the */       \
      /* element that matches this pattern. */                          \
      *topK = findPattern<DataType, BitDataType, IndexType>(                         \
        (DataType*) smem, data, sliceSize,                              \
        withinSliceStride, desired, desiredMask);                       \
      return;                                                           \
    }                                                                   \
                                                                        \
    if (count >= kToFind) {                                             \
      desired = Bitfield<BitDataType>::setBitfield(desired, i, digitPos, RADIX_BITS);          \
      desiredMask =                                                     \
        Bitfield<BitDataType>::setBitfield(desiredMask, RADIX_MASK, digitPos, RADIX_BITS);     \
                                                                        \
      /* The top-Kth element v must now be one such that: */            \
      /* (v & desiredMask == desired) */                                \
      /* but we haven't narrowed it down; we must check the next */     \
      /* least-significant digit */                                     \
      break;                                                            \
    }                                                                   \
                                                                        \
    kToFind -= count                                                    \

      if(Order) {
        // Process in descending order
#pragma unroll
        for(int i = RADIX_SIZE - 1; i >= 0; --i) {
          CHECK_RADIX(i);
        }
      }
      else {
        // Process in ascending order
#pragma unroll
        for(int i = 0; i < RADIX_SIZE; ++i) {
          CHECK_RADIX(i);
        }
      }
#undef CHECK_RADIX
    } // end digitPos for

    // There is no unique result, but there is a non-unique result
    // matching `desired` exactly
    *topK = TopKTypeConfig<DataType>::deconvert(desired);
  }

  // helpers2
  template <typename T>
  struct AddOp {
    __device__ __forceinline__ T operator()(T const &lhs, T const &rhs) {
      return lhs + rhs;
    }
  };

  __device__ __forceinline__ unsigned getLaneMaskLe() {
    unsigned mask;
    asm("mov.u32 %0, %%lanemask_le;" : "=r"(mask));
    return mask;
  }

  // Inclusive prefix sum for binary vars using intra-warp voting +
  // shared memory
  template <typename T, bool KillWARDependency, class BinaryFunction>
  __device__ void inclusiveBinaryPrefixScan(T* smem, bool in, T* out, BinaryFunction binop) {
    // Within-warp, we use warp voting.
    T vote = WARP_BALLOT(in);
    T index = __popc(getLaneMaskLe() & vote);
    T carry = __popc(vote);

    int warp = threadIdx.x / 32;

    // Per each warp, write out a value
    if(getLaneId() == 0) {
      smem[warp] = carry;
    }

    __syncthreads();

    // Sum across warps in one thread. This appears to be faster than a
    // warp shuffle scan for CC 3.0+
    if(threadIdx.x == 0) {
      int current = 0;
      for(int i = 0; i < blockDim.x / 32; ++i) {
        T v = smem[i];
        smem[i] = binop(smem[i], current);
        current = binop(current, v);
      }
    }

    __syncthreads();

    // load the carry from the preceding warp
    if(warp >= 1) {
      index = binop(index, smem[warp - 1]);
    }

    *out = index;

    if(KillWARDependency) {
      __syncthreads();
    }
  }

  // Exclusive prefix sum for binary vars using intra-warp voting +
  // shared memory
  template <typename T, bool KillWARDependency, class BinaryFunction>
  __device__ void exclusiveBinaryPrefixScan(T* smem, bool in, T* out, T* carry, BinaryFunction binop) {
    inclusiveBinaryPrefixScan<T, false, BinaryFunction>(smem, in, out, binop);

    // Inclusive to exclusive
    *out -= (T)in;

    // The outgoing carry for all threads is the last warp's sum
    *carry = smem[(blockDim.x / 32) - 1];

    if(KillWARDependency) {
      __syncthreads();
    }
  }

  // the kernal function
  template<typename T, typename IndexType, bool Order>
  __global__ void topk_kernel(T* input, T* out_val, IndexType* out_idx,
                 IndexType cur_size, IndexType k, IndexType outer_size, IndexType inner_size){
    // Indices are limited to integer fp precision, so counts can fit in
    // int32, regardless of IndexType
    __shared__ int smem[32]; // one per each warp, up to warp limit
    // in range, kind of like in the batch-size range (if flattened, that will be)
    IndexType slice = blockIdx.x;
    if(slice >= outer_size*inner_size) {
      return;
    }
    // prepare data
    T* inputSliceStart = input + slice/inner_size*inner_size*cur_size + slice%inner_size;
    T* topKSliceStart = out_val + slice/inner_size*inner_size*k + slice%inner_size;
    IndexType* indicesSliceStart = out_idx + slice/inner_size*inner_size*k + slice%inner_size;
    // Find the k-th highest element in our input
    T topKValue = (T)(0);
    radixSelect<T, typename TopKTypeConfig<T>::RadixType, IndexType, Order>(
      inputSliceStart, k, cur_size, inner_size, smem, &topKValue);

    // Every value that is strictly less/greater than `pattern`
    // (depending on sort dir) in sorted int format is in the top-K.
    // The top-K value itself might not be unique.
    //
    // Since there are a variable number of elements that we see that
    // are within the top-k, we don't know at what index to write out
    // the resulting values.
    // In order to get this, we perform an exclusive prefix sum of
    // `hasTopK`. This will return the resulting index into which we
    // need to write the result, if a thread has a result.

    // All threads need to participate in the loop and the prefix sum,
    // but not necessarily in the load; hence loop bounds being rounded
    // up to a multiple of the block dim.
    IndexType numIterations = THCRoundUp(cur_size, (IndexType)blockDim.x);
    IndexType writeIndexStart = 0;
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < cur_size);
      T v = inRange ? doLdg(&inputSliceStart[i * inner_size]) : (T)(0);
      bool hasTopK;
      if(Order) {   // Process in descending order
        hasTopK = inRange && (v > topKValue);
      }
      else {        // Process in ascending order
        hasTopK = inRange && (v < topKValue);
      }
      int index;
      int carry;
      exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());
      if(hasTopK) {
        int writeIndex = writeIndexStart + index;
        assert(writeIndex < k);

        IndexType topKOffset = writeIndex * inner_size;
        IndexType indexOffset = writeIndex * inner_size;

        topKSliceStart[topKOffset] = v;
        indicesSliceStart[indexOffset] = i + 0;
      }
      writeIndexStart += carry;
    }

    // We need to fill in the rest with actual == top-K values.
    // The number that we need is outputSliceSize -
    // writeIndexStart. There might be more than that number available,
    // in which case we have to choose the first seen set. We do this
    // via a prefix sum to calculate indices for writing results.
    assert(k >= writeIndexStart);
    IndexType topKRemaining = (k - writeIndexStart);
    for(IndexType i = threadIdx.x; i < numIterations; i += blockDim.x) {
      bool inRange = (i < cur_size);
      T v = inRange ? doLdg(&inputSliceStart[i * inner_size]) : (T)(0);
      bool hasTopK = inRange && (v == topKValue);
      int index;
      int carry;
      exclusiveBinaryPrefixScan<int, true>(smem, hasTopK, &index, &carry, AddOp<int>());
      if(hasTopK && index < topKRemaining) {
        int writeIndex = writeIndexStart + index;
        assert(writeIndex < k);

        IndexType topKOffset = writeIndex * inner_size;
        IndexType indexOffset = writeIndex * inner_size;

        topKSliceStart[topKOffset] = v;
        indicesSliceStart[indexOffset] = i + 0;
      }

      if(carry >= topKRemaining) {
        break;
      }
      topKRemaining -= carry;
      writeIndexStart += carry;
    }
    return;
  }

#undef RADIX_BITS
#undef RADIX_SIZE
#undef RADIX_MASK

  // final call for the topk (no sorting, only selecting topk)
  // -- (batch_size,cur_size,inner_size) -> (batch_size,k,inner_size)
  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k){
    // todo: check that inputs are valid
    //
    IndexType grid = outer_size*inner_size;
    IndexType block = (std::min(THCRoundUp(cur_size, (IndexType)(32)), (IndexType)(1024)));
    topk_kernel<T, IndexType, IsMax>
      <<<grid, block>>>(input, ouput_val, ouput_idx, cur_size, k, outer_size, inner_size);
    return cudaGetLastError();
  }
};

// using eigen, but only one-argmax
namespace impl_eigen{
  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
    Eigen::GpuDevice& my_device){
    // only argmax
    if(k != 1)
      throw std::runtime_error("Not implemented with EIGEN for k != 1.");
    const Eigen::array<Eigen::DenseIndex, 1> reduction_axis = {1};
    if(IsMax){
      (Eigen::TensorMap<Eigen::Tensor<IndexType, 2>>(ouput_idx, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).argmax(1);
      (Eigen::TensorMap<Eigen::Tensor<T, 2>>(ouput_val, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).maximum(reduction_axis);
    }
    else{
      (Eigen::TensorMap<Eigen::Tensor<IndexType, 2>>(ouput_idx, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).argmin(1);
      (Eigen::TensorMap<Eigen::Tensor<T, 2>>(ouput_val, inner_size, outer_size)).device(my_device)
        = (Eigen::TensorMap<Eigen::Tensor<T, 3>>(input, inner_size, cur_size, outer_size)).minimum(reduction_axis);
    }
    return cudaGetLastError();
  }
};

// tensorflow's implementation using max/min-heap
// - from tensorflow/tensorflow/core/kernels/topk_op_gpu.cu.cc
namespace impl_tf {

  enum class HeapType { kMinHeap, kMaxHeap };
  enum class PreferIndices { kLower, kHigher };   // prefer which index if equal

  template <typename T, typename IndexType>
  struct Entry {
    IndexType index;
    T value;
  };

  template <typename T, typename IndexType>
  struct LinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](IndexType index) const { return data[index]; }
    __device__ IndexType get_index(IndexType i) const { return data[i].index; }
    __device__ T get_value(IndexType i) const { return data[i].value; }
    Entry* const data;
  };

  template <typename T, typename IndexType>
  struct IndirectLinearData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](IndexType index) const { return data[index]; }
    __device__ IndexType get_index(IndexType i) const { return backing_data[data[i].index].index; }
    __device__ T get_value(IndexType i) const { return data[i].value; }
    Entry* const data;
    Entry* const backing_data;
  };

  template <typename T, typename IndexType>
  struct StridedData {
    typedef impl_tf::Entry<T, IndexType> Entry;
    __device__ Entry& operator[](IndexType index) const { return data[index * blockDim.x + threadIdx.x]; }
    __device__ IndexType get_index(IndexType i) const { return (*this)[i].index; }
    __device__ T get_value(IndexType i) const { return (*this)[i].value; }
    Entry* const data;
  };


  // A heap of Entry<T> that can either work as a min-heap or as a max-heap.
  template <HeapType heapType, PreferIndices preferIndices, template <typename,typename> class Data, typename T, typename IndexType>
  struct IndexedHeap {
    typedef typename Data<T, IndexType>::Entry Entry;
    const Data<T, IndexType> data;

    // indicating whether left should be prior to right
    __device__ bool is_above(IndexType left, IndexType right) {
      T left_value = data.get_value(left);
      T right_value = data.get_value(right);
      if(left_value == right_value) {
        if(preferIndices == PreferIndices::kLower) {
          return data.get_index(left) < data.get_index(right);
        }
        else {
          return data.get_index(left) > data.get_index(right);
        }
      }
      if(heapType == HeapType::kMinHeap) {
        return left_value < right_value;
      }
      else {
        return left_value > right_value;
      }
    }

    // assign one entry
    __device__ void assign(IndexType i, const Entry& entry) { data[i] = entry; }

    // swap
    __device__ void swap(IndexType a, IndexType b) {
      auto tmp = data[b];
      data[b] = data[a];
      data[a] = tmp;
    }

    // upward from i
    __device__ void push_up(IndexType i) {
      IndexType child = i;
      IndexType parent;
      for(; child > 0; child = parent) {
        parent = (child - 1) / 2;
        if(!is_above(child, parent)) {
          // Heap property satisfied.
          break;
        }
        swap(child, parent);
      }
    }

    __device__ void push_root_down(IndexType k) { push_down(0, k); }

    // MAX-HEAPIFY in Cormen, k is the range
    __device__ void push_down(IndexType node, IndexType k) {
      while(true) {
        const IndexType left = 2 * node + 1;
        const IndexType right = left + 1;
        IndexType smallest = node;
        if(left < k && is_above(left, smallest)) {
          smallest = left;
        }
        if(right < k && is_above(right, smallest)) {
          smallest = right;
        }
        if(smallest == node) {
          break;
        }
        swap(smallest, node);
        node = smallest;
      }
    }

    // BUILD-MAX-HEAPIFY in Cormen
    __device__ void build(IndexType k) {
      for(IndexType node = (k - 1) / 2; node >= 0; node--) {
        push_down(node, k);
      }
    }

    // HEAP-EXTRACT-MAX in Cormen
    __device__ void remove_root(IndexType k) {
      data[0] = data[k - 1];
      push_root_down(k - 1);
    }

    // in-place HEAPSORT in Cormen (turn minHeap to max-sorting)
    // This method destroys the heap property.
    __device__ void sort(IndexType k) {
      for(IndexType slot = k - 1; slot > 0; slot--) {
        // This is like remove_root but we insert the element at the end.
        swap(slot, 0);
        // Heap is now an element smaller.
        push_root_down(/*k=*/slot);
      }
    }

    __device__ void replace_root(const Entry& entry, IndexType k) {
      data[0] = entry;
      push_root_down(k);
    }

    __device__ const Entry& root() { return data[0]; }
  };

  template <HeapType heapType, PreferIndices preferIndices, template <typename,typename> class Data, typename T, typename IndexType>
  __device__ IndexedHeap<heapType, preferIndices, Data, T, IndexType> make_indexed_heap(
    typename Data<T, IndexType>::Entry* data) {
    return IndexedHeap<heapType, preferIndices, Data, T, IndexType>{Data<T, IndexType>{data}};
  }

  // heapTopK walks over [input, input+length) with `step_size` stride starting at
  // `start_index`.
  // It builds a top-`k` heap that is stored in `heap_entries` using `Accessor` to
  // access elements in `heap_entries`. If sorted=true, the elements will be
  // sorted at the end.
  template <typename T, typename IndexType, template <typename, typename> class Data = LinearData>
  __device__ void heapTopK(const T* __restrict__ input, IndexType length, IndexType k, Entry<T, IndexType>* __restrict__ heap_entries,
    bool sorted = false, IndexType start_index = 0, IndexType step_size = 1) {
    // this should be restricted previously
    //assert(k <= (length-start_index+step_size-1)/step_size);
    // the min value as the threshold
    auto heap = make_indexed_heap<HeapType::kMinHeap, PreferIndices::kHigher, Data, T, IndexType>(heap_entries);
    IndexType heap_end_index = start_index + k * step_size;
    if(heap_end_index > length) {
      heap_end_index = length;
    }
    // Initialize the min-heap with the first k ones.
    for(IndexType index = start_index, slot = 0; index < heap_end_index; index += step_size, slot++) {
      heap.assign(slot, {index, input[index]});
    }
    heap.build(k);
    // Now iterate over the remaining items.
    // If an item is smaller than the min element, it is not amongst the top k.
    // Otherwise, replace the min element with it and push upwards.
    for(IndexType index = heap_end_index; index < length; index += step_size) {
      // We prefer elements with lower indices. This is given here.
      // Later elements automatically have higher indices, so can be discarded.
      if(input[index] > heap.root().value) {
        // This element should replace the min.
        heap.replace_root({index, input[index]}, k);
      }
    }
    // Sort if wanted.
    if(sorted) {
      heap.sort(k);
    }
  }

  // mergeShards performs a top-k merge on `num_shards` many sorted streams that
  // are sorted and stored in `entries` in a strided way:
  // |s_1 1st|s_2 1st|...s_{num_shards} 1st|s_1 2nd|s_2 2nd|...
  // The overall top k elements are written to `top_k_values` and their indices
  // to top_k_indices.
  // `top_k_heap` is used as temporary storage for the merge heap.
  template <typename T, typename IndexType>
  __device__ void mergeShards(IndexType num_shards, IndexType k,
    Entry<T, IndexType>* __restrict__ entries, Entry<T, IndexType>* __restrict__ top_k_heap,
    T* top_k_values, IndexType* top_k_indices) {
    // If k < num_shards, we can use a min-heap with k elements to get the top k
    // of the sorted blocks.
    // If k > num_shards, we can initialize a min-heap with the top element from
    // each sorted block.
    const IndexType heap_size = k < num_shards ? k : num_shards;

    // TODO: the heaps could have garbage if too many shards or too few length
    // Min-heap part.
    {
      auto min_heap = IndexedHeap<HeapType::kMinHeap, PreferIndices::kHigher,
        IndirectLinearData, T, IndexType>{IndirectLinearData<T, IndexType>{top_k_heap, entries}};
      // Initialize the heap as a min-heap.
      for(IndexType slot = 0; slot < heap_size; slot++) {
        min_heap.assign(slot, {slot, entries[slot].value});
      }
      min_heap.build(heap_size);

      // Now perform top k with the remaining shards (if num_shards > heap_size).
      for(IndexType shard = heap_size; shard < num_shards; shard++) {
        const auto entry = entries[shard];
        const auto root = min_heap.root();
        if(entry.value < root.value) {
          continue;
        }
        if(entry.value == root.value &&
          entry.index > entries[root.index].index) {
          continue;
        }
        // This element should replace the min.
        min_heap.replace_root({shard, entry.value}, heap_size);
      }
    }

    // Max-part.
    {
      // Turn the min-heap into a max-heap in-place.
      auto max_heap = IndexedHeap<HeapType::kMaxHeap, PreferIndices::kLower,
        IndirectLinearData, T, IndexType>{
        IndirectLinearData<T, IndexType>{top_k_heap, entries}};
      // Heapify into a max heap.
      max_heap.build(heap_size);

      // Now extract the minimum k-1 times.
      // k is treated specially.
      const IndexType last_k = k - 1;
      for(IndexType rank = 0; rank < last_k; rank++) {
        const Entry<T, IndexType>& max_element = max_heap.root();
        top_k_values[rank] = max_element.value;
        IndexType shard_index = max_element.index;
        top_k_indices[rank] = entries[shard_index].index;
        IndexType next_shard_index = shard_index + num_shards;
        // For rank < k-1, each top k heap still contains at least 1 element,
        // so we can draw a replacement.
        max_heap.replace_root({next_shard_index, entries[next_shard_index].value},
          heap_size);
      }

      // rank == last_k.
      const Entry<T, IndexType>& max_element = max_heap.root();
      top_k_values[last_k] = max_element.value;
      IndexType shard_index = max_element.index;
      top_k_indices[last_k] = entries[shard_index].index;
    }
  }

  extern __shared__ char shared_memory[];

  template <typename T, typename IndexType>
  __global__ void TopKKernel(const T* input, IndexType length, IndexType k,
    T* output, IndexType* indices) {
    const IndexType batch_index = blockIdx.x;
    const T* batch_input = input + batch_index * length;

    const IndexType thread_index = threadIdx.x;
    const IndexType thread_count = blockDim.x;

    Entry<T, IndexType>* shared_entries = (Entry<T, IndexType>*)shared_memory;

    // heap-select with strided elements
    heapTopK<T, IndexType, StridedData>(batch_input, length, k, shared_entries, true,
      thread_index, thread_count);

    __syncthreads();
    if(thread_index == 0) {
      const IndexType offset = batch_index * k;
      auto batch_output = output + offset;
      auto batch_indices = indices + offset;
      Entry<T, IndexType>* top_k_heap = shared_entries + thread_count * k;

      // TODO(blackhc): Erich says: Performance can likely be improved
      // significantly by having the merge be done by multiple threads rather than
      // just one.  ModernGPU has some nice primitives that could help with this.
      mergeShards(thread_count, k, shared_entries, top_k_heap, batch_output,
        batch_indices);
    }
  }

  template<typename T, typename IndexType>
  IndexType get_num_shards(IndexType k, IndexType length){
    // This code assumes that k is small enough that the computation
    // fits inside shared memory (hard coded to 48KB).  In practice this
    // means k <= 3072 for T=float/int32 and k <= 2048 for T=double/int64.
    // The calculation is:
    //   shared_memory_size / (2 * (sizeof(int) + sizeof(T))) < k.
    // Use as many shards as possible.
    IndexType num_shards;
    constexpr IndexType shared_memory_size = 48 << 10;  // 48 KB
    const IndexType heap_size = k * sizeof(Entry<T, IndexType>);
    // shared_memory_size = (num_shards + 1) * heap_size <=>
    num_shards = shared_memory_size / heap_size - 1;
    if(num_shards <= 0) {
      num_shards = 1;
    }
    auto shard_size = length / num_shards;
    auto min_shard_size = 2 * k;
    if(shard_size < min_shard_size) {
      num_shards = length / min_shard_size;
    }
    if(num_shards <= 0) {
      num_shards = 1;
    }
    else if(num_shards > 1024) {
      num_shards = 1024;
    }
    //std::cout << "Auto-setting num_shards=" << num_shards << std::endl;
    return num_shards;
  }

  template<typename T, typename IndexType, bool IsMax>
  cudaError topk(T* input, T* ouput_val, IndexType* ouput_idx,
    IndexType outer_size, IndexType inner_size, IndexType cur_size, IndexType k,
    IndexType num_shards = 0){
    // stricted, but could be extended
    if(!IsMax)
      throw std::runtime_error("Not implemented with tf for n-argmin.");
    if(inner_size != 1)
      throw std::runtime_error("Not implemented with tf for strided version.");
    //
    if(num_shards <= 0)
      num_shards = get_num_shards<T, IndexType>(k, cur_size);
    //
    if(cur_size < num_shards*k)
      throw std::runtime_error("Given too many shards for the current situation.");
    auto shared_memory_size = (num_shards + 1) * k * sizeof(Entry<T, IndexType>);   //double that for possible different sizes of host/device
    TopKKernel<<<outer_size, num_shards, shared_memory_size >>>(input, cur_size, k, ouput_val, ouput_idx);
    return cudaGetLastError();
  }
};

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

//#define DEBUG
template<typename T, typename IndexType>
void check(IndexType NUM_OUTPUT, IndexType NUM_BATCH, IndexType K, 
  vector<unordered_set<IndexType>>& answer, T* v_input, T* v_val, IndexType* v_idx){
  int bad_count = 0;
  for(int i = 0; i < NUM_BATCH; i++){
#ifdef DEBUG
    auto cur_input = v_input + i*NUM_OUTPUT;
    auto cur_val = v_val + i*K;
#endif // DEBUG
    auto cur_idx = v_idx + i*K;
    const unordered_set<IndexType>& gold = answer[i];
    unordered_set<IndexType> pred;
    for(int j = 0; j < K; j++)
      pred.insert(cur_idx[j]);
    if(gold != pred){
      bad_count++;
#ifdef DEBUG
      // sort the results
      using PairType = std::pair<IndexType,T>;
      vector<PairType> gold_ranks, pred_ranks;
      for(auto idx : gold)
        gold_ranks.push_back(std::make_pair(idx, cur_input[idx]));
      for(int j = 0; j < K; j++)
        pred_ranks.push_back(std::make_pair(cur_idx[j], cur_val[j]));
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
    cout << "Check unequal of batches of " << bad_count << endl;
#endif
}


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

// testing the most common case: dimension-2 k-argmax
template<typename T, typename IndexType>
void test2(IndexType NUM_OUTPUT, IndexType NUM_BATCH, IndexType K, IndexType NUM_STEP){
  IndexType NUM_STRIDE = 1;
  // allocating
  size_t size_input = NUM_OUTPUT * NUM_BATCH * sizeof(T);
  size_t size_val = K * NUM_BATCH * sizeof(T);
  size_t size_idx = K * NUM_BATCH * sizeof(IndexType);
  auto v_input = new T[NUM_OUTPUT*NUM_BATCH];
  auto v_val = new T[K*NUM_BATCH];
  auto v_idx = new IndexType[K*NUM_BATCH];
#ifdef DEBUG
  for(int i = 0; i < NUM_OUTPUT*NUM_BATCH; i++)
    v_input[i] = T(i%NUM_OUTPUT);
#else
  for(int i = 0; i < NUM_OUTPUT*NUM_BATCH; i++)
    v_input[i] = T(get_rand(1));
#endif
  //
  cout << "--- Start a test: T=" << typeid(T).name() << " IndexType=" << typeid(IndexType).name() << ' '
       << NUM_OUTPUT << ' ' << NUM_BATCH << ' ' << K << ' ' << NUM_STEP << endl;
  // sort on cpu as the gold answer
  //
  vector<unordered_set<IndexType>> answers(NUM_BATCH);
  {
    cout << "== Test with cpu-sorting" << endl;
    auto t1 = clock();
    using PairType = std::pair<T,IndexType>;
    for(int i = 0; i < NUM_BATCH; i++){
      auto cur_input = v_input + i*NUM_OUTPUT;
      vector<PairType> result;
      for(int j = 0; j < NUM_OUTPUT; j++)
        result.push_back(std::make_pair(cur_input[j], j));
      std::sort(result.begin(), result.end(), [](const PairType& a, const PairType& b){ return a.first>b.first;});
      for(int j = 0; j < K; j++)
        answers[i].insert(result[j].second);
    }
    auto t2 = clock();
    //
    double time_once = double(t2 - t1) / CLOCKS_PER_SEC;
    cout << "Time for sort is " << time_once << ' ' << time_once*NUM_STEP << endl;
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
  /*
  try{
    cout << "== Test with impl_eigen" << endl;
    double timek = 0;
    double timec = 0;
    // calculate
    for(int i = 0; i < NUM_STEP; i++){
      auto t1 = clock();
      CUDA_CHECK(impl_eigen::topk<T,IndexType,true>(cv_input, cv_val, cv_idx, NUM_BATCH, NUM_STRIDE, NUM_OUTPUT, K, *edevice));
      auto t2 = clock();
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
      auto t3 = clock();
      timek += (t2 - t1);
      timec += (t3 - t2);
    }
    check(NUM_OUTPUT, NUM_BATCH, K, answers, v_input, v_val, v_idx);
    cout << "Time for eigen is " << timek / CLOCKS_PER_SEC << "/" << timec / CLOCKS_PER_SEC << endl;
  }
  catch(...){ cout << "error" << endl; }
  */
  //
  try{
    cout << "== Test with impl_tf, auto-shard is " << impl_tf::get_num_shards<T, IndexType>(K, NUM_OUTPUT) << endl;
    for(IndexType num_shards : {0, 64, 128, 256, 512}){
      double timek = 0;
      double timec = 0;
      // calculate
      for(int i = 0; i < NUM_STEP; i++){
        auto t1 = clock();
        CUDA_CHECK(impl_tf::topk<T, IndexType, true>(cv_input, cv_val, cv_idx, NUM_BATCH, NUM_STRIDE, NUM_OUTPUT, K, num_shards));
        auto t2 = clock();
        // copy back
        cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
        cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
        auto t3 = clock();
        timek += (t2 - t1);
        timec += (t3 - t2);
      }
      check(NUM_OUTPUT, NUM_BATCH, K, answers, v_input, v_val, v_idx);
      cout << "Time for tf-" << num_shards << " is " << timek / CLOCKS_PER_SEC << "/" << timec / CLOCKS_PER_SEC << endl;
    }
  }
  catch(...){ cout << "error" << endl; }
  //
  try{
    cout << "== Test with impl_tr" << endl;
    double timek = 0;
    double timec = 0;
    // calculate
    for(int i = 0; i < NUM_STEP; i++){
      auto t1 = clock();
      CUDA_CHECK(impl_tr::topk<T, IndexType, true>(cv_input, cv_val, cv_idx, NUM_BATCH, NUM_STRIDE, NUM_OUTPUT, K));
      auto t2 = clock();
      // copy back
      cudaMemcpy(v_val, cv_val, size_val, cudaMemcpyDeviceToHost);
      cudaMemcpy(v_idx, cv_idx, size_idx, cudaMemcpyDeviceToHost);
      auto t3 = clock();
      timek += (t2 - t1);
      timec += (t3 - t2);
    }
    check(NUM_OUTPUT, NUM_BATCH, K, answers, v_input, v_val, v_idx);
    auto t2 = clock();
    cout << "Time for tr is " << timek / CLOCKS_PER_SEC << "/" << timec / CLOCKS_PER_SEC << endl;
  }
  catch(...){ cout << "error" << endl; }
  // free
  cudaFree(cv_input);
  cudaFree(cv_val);
  cudaFree(cv_idx);
  cout << endl;
  return;
}

int main()
{
  setup(7);
  //
  test2<float, int>(40, 1, 4, 100);
  test2<float, int>(50000, 80, 1, 100);
  test2<float, int>(50000, 80, 2, 100);
  test2<float, int>(50000, 80, 4, 100);
  test2<float, int>(50000, 80, 8, 100);
  test2<float, int>(50000, 80, 12, 100);
  test2<float, Eigen::DenseIndex>(40, 1, 4, 100);
  test2<float, Eigen::DenseIndex>(50000, 80, 1, 100);
  test2<float, Eigen::DenseIndex>(50000, 80, 2, 100);
  test2<float, Eigen::DenseIndex>(50000, 80, 4, 100);
  test2<float, Eigen::DenseIndex>(50000, 80, 8, 100);
  test2<float, Eigen::DenseIndex>(50000, 80, 12, 100);
  test2<double, int>(40, 1, 4, 100);
  test2<double, int>(50000, 80, 1, 100);
  test2<double, int>(50000, 80, 2, 100);
  test2<double, int>(50000, 80, 4, 100);
  test2<double, int>(50000, 80, 8, 100);
  test2<double, int>(50000, 80, 12, 100);
  test2<double, Eigen::DenseIndex>(40, 1, 4, 100);
  test2<double, Eigen::DenseIndex>(50000, 80, 1, 100);
  test2<double, Eigen::DenseIndex>(50000, 80, 2, 100);
  test2<double, Eigen::DenseIndex>(50000, 80, 4, 100);
  test2<double, Eigen::DenseIndex>(50000, 80, 8, 100);
  test2<double, Eigen::DenseIndex>(50000, 80, 12, 100);
  
  return 0;
}

// compile: nvcc -std=c++11 -DEIGEN_USE_GPU -I../libs/eigen-eigen-5a0156e40feb-334/ ./tk.cu

