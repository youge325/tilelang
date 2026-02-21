# 通用矩阵-向量乘法（GEMV）
===========================================

<div style="text-align: left;">
    <em>贡献者：</em> <a href="https://github.com/botbw">@botbw</a>
</div>

:::{warning}
   本文档仍处于**实验阶段**，可能不完整。
   非常欢迎提出建议和改进——请提交 PR！
:::

:::{tip}
示例代码可在 `examples/gemv/example_gemv.py` 中找到。
:::

通用矩阵-向量乘法（GEMV）可以视为通用矩阵-矩阵乘法（GEMM）的特殊情况。它在深度学习中扮演着关键角色，尤其是在大语言模型的推理阶段。在本教程中，我们将使用 `TileLang` 从线程级视角逐步优化 GEMV。

## Triton 实现
在实现 GEMV 内核时，您可能会从使用 `Triton` 等工具的高层方法开始。

一个简单的 Triton GEMV 内核可能如下所示：
```python
@triton.jit
def _gemv_naive(
    x_ptr, A_ptr, y_ptr,
    N, K,
    BLOCK_SIZE_K: tl.constexpr,
):
    n = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    mask = offs_k < K
    a_ptrs = A_ptr + n * K + offs_k
    a_vals = tl.load(a_ptrs, mask=mask, other=0.0)
    x_vals = tl.load(x_ptr + offs_k, mask=mask, other=0.0)
    dot = tl.sum(a_vals * x_vals, axis=0)
    tl.store(y_ptr + n, dot)
```

`Triton` 使用起来很直接，因为它在块级别操作。然而，这种方法可能不允许进行细粒度的线程级优化。在本教程中，我们将演示如何在 `TileLang` 中编写优化的 GEMV 内核，以暴露更多底层控制。

## TileLang 中的朴素实现
如果您对 CUDA C 有基本了解，那么通过改编 GEMM 的分块策略来开始编写朴素的 GEMV 内核是很自然的。您可以将 GEMV 视为 `(1, k) * (k, n)` 的 GEMM。以下是一个简单的示例：

```python
def naive_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N)) as bn:
            tn = T.get_thread_binding(0)  # tn = threadIdx.x
            A_shared = T.alloc_shared((BLOCK_K,), dtype)
            B_shared = T.alloc_shared((BLOCK_N, BLOCK_K), dtype)
            C_reg = T.alloc_local((1,), accum_dtype)
            T.clear(C_reg)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for tk in T.serial(BLOCK_K):
                    A_shared[tk] = A[bk * BLOCK_K + tk]
                    B_shared[tn, tk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                for tk in T.serial(BLOCK_K):
                    C_reg[0] += A_shared[tk].astype(accum_dtype) * B_shared[tn,
                                                                            tk].astype(accum_dtype)
            C[bn * BLOCK_N + tn] = C_reg[0]

    return main
```

您的内核将由 `TileLang` 编译为 CUDA（在 `~/.tilelang/cache` 中）：

```C++
extern "C" __global__ void __launch_bounds__(256, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  extern __shared__ __align__(1024) uchar buf_dyn_shmem[];
  float C_reg[1];
  __shared__ uint64_t _mbarrier[2];
  if (((int)threadIdx.x) == 0) {
    tl::mbarrier_init(_mbarrier[0], 128);
    tl::mbarrier_init(_mbarrier[1], 128);
  }
  __syncthreads();
  if (128 <= ((int)threadIdx.x)) {
    tl::warpgroup_reg_dealloc<24>();
    for (int bk = 0; bk < 8; ++bk) {
      tl::mbarrier_wait(_mbarrier[1], ((bk & 1) ^ 1));
      for (int tk = 0; tk < 128; ++tk) {
        ((half_t*)buf_dyn_shmem)[tk] = A[((bk * 128) + tk)];
        ((half_t*)buf_dyn_shmem)[(((((int)threadIdx.x) * 128) + tk) - 16256)] = B[(((((((int)blockIdx.x) * 131072) + (((int)threadIdx.x) * 1024)) + (bk * 128)) + tk) - 131072)];
      }
      tl::fence_proxy_async();
      tl::mbarrier_cp_async_arrive(_mbarrier[0]);
      tl::mbarrier_arrive(_mbarrier[0]);
    }
  } else {
    tl::warpgroup_reg_alloc<240>();
    C_reg[0] = 0.000000e+00f;
    for (int bk_1 = 0; bk_1 < 8; ++bk_1) {
      tl::mbarrier_wait(_mbarrier[0], (bk_1 & 1));
      for (int tk_1 = 0; tk_1 < 128; ++tk_1) {
        C_reg[0] = (C_reg[0] + (((float)((half_t*)buf_dyn_shmem)[tk_1]) * ((float)((half_t*)buf_dyn_shmem)[(((((int)threadIdx.x) * 128) + tk_1) + 128)])));
      }
      tl::fence_proxy_async();
      tl::mbarrier_arrive(_mbarrier[1]);
    }
    C[((((int)blockIdx.x) * 128) + ((int)threadIdx.x))] = ((half_t)C_reg[0]);
  }
}
```

在这个设计中，一个块内的前 128 个线程充当数据生产者，后 128 个线程充当消费者（假设为 1D 块）。

在这个级别上，我们从 GPU 获得的计算能力非常有限，延迟约为 **~0.17 ms**，而 torch/cuBLAS 约为 **~0.008 ms**，大约慢 20 倍。

## 更多并发

为了进一步提高内核的并发性，我们可以利用更细粒度的线程级并行。与其将每个线程分配计算 C 中的单个输出元素，不如沿 K 维度引入并行。每个线程计算部分累加，然后合并这些部分结果。这种方法需要 CUDA 中的 `atomicAdd` 等原语。

以下是简化版本：
```python
def naive_splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, BLOCK_K)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((1,), dtype)
            B_local = T.alloc_local((1,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                A_local[0] = A[bk * BLOCK_K + tk]
                B_local[0] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk]
                C_accum[0] += A_local[0].astype(accum_dtype) * B_local[0].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

通过在 K 维度引入并行，我们的内核现在达到了 **~0.024 ms**，有所改善，但仍未达到 torch/cuBLAS 的水平。

### 自定义 K 维度的并行度
如果您的 K 维度很大，可以通过引入 `reduce_threads` 参数进一步自定义每个线程处理的元素数量。这样每个线程在每次迭代中处理多个元素：

```python
def splitk_gemv(
    N: int,
    K: int,
    BLOCK_N: int,
    BLOCK_K: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    TILE_K = T.ceildiv(BLOCK_K, reduce_threads)

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.serial(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

## 向量化读取

GEMV 的计算密度低于 GEMM，因此计算强度和内存吞吐量将成为优化瓶颈。一种有效的策略是使用向量化的加载/存储操作（如 `float2`、`float4`）。在 `TileLang` 中，您可以通过 `T.vectorized` 指定向量化操作：

```python
def splitk_gemv_vectorized(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype)
            C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]

    return main
```

使用向量化读取后，内核延迟降至 **~0.0084 ms**，接近 cuBLAS 的性能。

## 使用 `tvm_thread_allreduce` 替代 `atomicAdd`

[`tvm_thread_allreduce`](https://tvm.apache.org/docs/reference/api/python/tir/tir.html#tvm.tir.tvm_thread_allreduce) 在跨多个线程进行全规约时实现了优化，应该优于我们简单的共享内存 + `atomicAdd` 方案：

```python
def splitk_gemv_vectorized_tvm(
    N: int,
    K: int,
    BLOCK_N: int,
    reduce_threads: int,
    dtype: str = "float16",
    accum_dtype: str = "float",
):
    MAX_TRANSACTION_SIZE_IN_BITS = 128
    TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
    BLOCK_K = reduce_threads * TILE_K

    @T.prim_func
    def main(
            A: T.Buffer((K,), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype)
            B_local = T.alloc_local((TILE_K,), dtype)
            C_accum = T.alloc_local((1,), accum_dtype)

            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for k in T.vectorized(TILE_K):
                    A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                    B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                for k in T.serial(TILE_K):
                    C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
            C_reduced = T.alloc_local((1,), accum_dtype)
            with T.attr(
                    T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                    "reduce_scope",
                    T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        C_accum[0],
                        True,
                        C_reduced[0],
                        tk,
                        dtype="handle",
                    ))

            C[bn * BLOCK_N + tn] = C_reduced[0]

    return main
```

通过这一优化，内核延迟从 **~0.0084 ms** 降至 **~0.0069 ms**，快于 torch/cuBLAS！

## 自动调优

`BLOCK_N`、`BLOCK_K`、`reduce_threads` 是内核中的超参数，可以通过调优来提升性能。我们可以使用 `tilelang.autotune` 功能来自动搜索最优配置：

```python
def get_best_config(N, K):

    def get_configs():
        BLOCK_N = [2, 4, 8, 32, 64, 128]
        reduce_threads = [4, 8, 32]
        _configs = list(itertools.product(
            BLOCK_N,
            reduce_threads,
        ))
        configs = [{
            "BLOCK_N": c[0],
            "reduce_threads": c[1],
        } for c in _configs]
        return configs

    @autotune(
        configs=get_configs(),
        warmup=3,
        rep=20,
    )
    @jit(
        out_idx=[-1],
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        target="auto",
    )
    def kernel(
        BLOCK_N=None,
        reduce_threads=None,
    ):
        dtype = "float16"
        accum_dtype = "float"
        MAX_TRANSACTION_SIZE_IN_BITS = 128
        TILE_K = MAX_TRANSACTION_SIZE_IN_BITS // DataType(dtype).bits
        BLOCK_K = reduce_threads * TILE_K

        @T.prim_func
        def main(
                A: T.Buffer((K,), dtype),
                B: T.Buffer((N, K), dtype),
                C: T.Buffer((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
                tn = T.get_thread_binding(0)
                tk = T.get_thread_binding(1)
                A_local = T.alloc_local((TILE_K,), dtype)
                B_local = T.alloc_local((TILE_K,), dtype)
                C_accum = T.alloc_local((1,), accum_dtype)

                T.clear(C_accum)
                for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                    for k in T.vectorized(TILE_K):
                        A_local[k] = A[bk * BLOCK_K + tk * TILE_K + k]
                        B_local[k] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + k]
                    for k in T.serial(TILE_K):
                        C_accum[0] += A_local[k].astype(accum_dtype) * B_local[k].astype(accum_dtype)
                C_reduced = T.alloc_local((1,), accum_dtype)
                with T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                ):
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_accum[0],
                            True,
                            C_reduced[0],
                            tk,
                            dtype="handle",
                        ))

                C[bn * BLOCK_N + tn] = C_reduced[0]

        return main

    return kernel()
```

经过自动调优，我们的内核达到了 **~0.0067 ms**，最终生成的 CUDA 内核可能如下所示：

```C++
extern "C" __global__ void __launch_bounds__(64, 1) main_kernel(half_t* __restrict__ A, half_t* __restrict__ B, half_t* __restrict__ C) {
  float C_accum[1];
  half_t A_local[8];
  half_t B_local[8];
  __shared__ float red_buf0[64];
  C_accum[0] = 0.000000e+00f;
  for (int bk = 0; bk < 4; ++bk) {
    *(uint4*)(A_local + 0) = *(uint4*)(A + ((bk * 256) + (((int)threadIdx.y) * 8)));
    *(uint4*)(B_local + 0) = *(uint4*)(B + ((((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 1024)) + (bk * 256)) + (((int)threadIdx.y) * 8)));
    for (int k = 0; k < 8; ++k) {
      C_accum[0] = (C_accum[0] + (((float)A_local[k]) * ((float)B_local[k])));
    }
  }
  tl::fence_proxy_async();
  __syncthreads();
  red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = C_accum[0];
  __syncthreads();
  if (((int)threadIdx.y) < 16) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 16)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 8) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 8)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 4) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 4)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 2) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 2)]);
  }
  __syncthreads();
  if (((int)threadIdx.y) < 1) {
    red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] = (red_buf0[((((int)threadIdx.x) * 32) + ((int)threadIdx.y))] + red_buf0[(((((int)threadIdx.x) * 32) + ((int)threadIdx.y)) + 1)]);
  }
  __syncthreads();
  C[((((int)blockIdx.x) * 2) + ((int)threadIdx.x))] = ((half_t)red_buf0[(((int)threadIdx.x) * 32)]);
}
```

这与我们的 `TileLang` 程序密切对应，必要的同步和底层优化已自动插入。

## 结论

### Hopper GPU 上的基准测试表

| 内核名称   | 延迟   |
|------------|------------|
| torch/cuBLAS | 0.00784 ms |
| Triton | 0.00773 ms |
| naive_gemv | 0.16607 ms |
| splitk_gemv | 0.02419 ms |
| splitk_gemv_vectorized | 0.00809 ms |
| splitk_gemv_vectorized_tvm | 0.00675 ms |

Triton 耗时：0.0077344514429569244
在本教程中，我们实现了一个简单的 GEMV 内核，并了解到 `TileLang` 向用户暴露了底层控制能力，如线程级编程和 CUDA 原语。
