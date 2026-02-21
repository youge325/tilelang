# 逐元素算子

<div style="text-align: left;">
    <em>作者：</em> <a href="https://github.com/chenghuaWang">Chenghua Wang</a>
</div>

:::{warning}
:class: myclass1 myclass2
:name: a-tip-reference

   本文档仍处于**实验阶段**，可能不完整。
   非常欢迎提出建议和改进——请提交 PR！
:::

逐元素算子在深度学习中被广泛使用，通常也是初学者探索并行编程时遇到的第一个示例。本教程将分析使用 TileLang 实现的几种逐元素加法算子，并将它们与对应的 CUDA 实现进行比较。在本教程结束时，您将学到：

1. 如何使用 TileLang 实现逐元素算子。
2. 如何编译具有动态形状的算子。
3. TileLang 如何处理边界相关问题。
4. 用 TileLang 实现的算子与用 CUDA/CuTe 实现的算子之间的异同。

请注意，本教程不会深入探讨 TileLang 的设计原理。要更全面地了解 TileLang，建议参阅[概述](../get_started/overview.md)。

## TileLang 中的逐元素加法

```python
def elementwise_add(N, threads=256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads), threads=threads) as (b_x):
            # 向量加法
            for i in T.Parallel(threads):
                C[b_x * threads + i] = A[b_x * threads + i] + B[b_x * threads + i]

    return main
```

TileLang 内核的所有逻辑必须在 `T.Kernel(...)` 作用域内实现。在本例中，初始化 `T.kernel(...)` 需要指定网格大小和每个块的线程数。返回值 `bx` 对应 CUDA 中的 `blockIdx.x`。在以上实现中，`T.Parallel` 用于处理分配给该块的数据 tile（大小为 `1 x threads`）进行计算。

熟悉 CUDA 编程的读者可能会问 `threadIdx` 在哪里。请注意，`T.Kernel` 内部的代码运行在**块级别（block level）**而非**线程级别（thread level）**。在本例中，您只需关注定义块级逻辑。在编译过程中，TileLang 会自动将计算映射到对应的线程并进行进一步优化。TileLang 生成的优化代码可能与精心手工编写的计算逻辑非常接近，如第 2 节中的具体示例所示。虽然 TileLang 也支持线程级编程语义，但这将在后续讨论中介绍。

可以使用以下代码编译程序：

```python
program = elementwise_add(1024, threads=256, dtype=T.bfloat16)
kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
```
启动内核非常简单，只需像函数一样直接调用：

```python
C = kernel(A, B)
```

向量加法运算也可以扩展到二维情况，两种实现在实际中展现出相当的效率。以下是测试部分的一个示例供读者参考：[示例](https://github.com/tile-ai/tilelang/blob/main/testing/python/kernel/test_tilelang_kernel_element_wise_add.py)。该内核的代码如下：

```python
import tilelang.language as T
def elementwise_add(
    M,
    N,
    block_M,
    block_N,
    in_dtype,
    out_dtype,
    threads,
):
    @T.prim_func
    def main(
            A: T.Tensor((M, N), in_dtype),
            B: T.Tensor((M, N), in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            start_x = bx * block_N
            start_y = by * block_M

            for (local_y, local_x) in T.Parallel(block_M, block_N):
                y = start_y + local_y
                x = start_x + local_x

                C[y, x] = A[y, x] + B[y, x]

    return main
```

### 如何编译具有动态形状的算子？

在上面的编译过程中，使用了固定形状。然而在实际使用中，我们经常希望内核支持动态形状。那么如何在 TileLang 中编译支持动态形状的内核呢？在 TileLang 中，我们可以用动态符号值替换目标大小，使该维度变为动态。以下示例说明了这一点：

```python
program = elementwise_add(T.dynamic("N"), threads=256, dtype=T.bfloat16)
kernel = tilelang.compile(program, out_idx=-1, target="cuda", execution_backend="cython")
```

生成的内核 CUDA 代码将在 `bfloat16_t* __restrict__ A`、`bfloat16_t* __restrict__ B` 和 `bfloat16_t* __restrict__ C` 参数之后包含一个额外的 `int N` 参数。

### TileLang 如何处理边界相关问题

TileLang 会自动加入边界检查条件；但这会有一定的代价。这些边界条件可能会阻止 TileLang 执行更高级的优化。我将提前介绍下一节的一个示例。相应代码也在下方提供，但请注意它涉及对应的 CUDA 代码。建议读者先阅读下一节再回到这段内容，以获得更清晰的理解。

当编译以下示例时，让我们将 `N` 设置为 2047：

```python
def elementwise_add(N, num_per_thread=8, threads=256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as (b_x):
            # 向量加法
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (b_x * threads + i) * num_per_thread
                C[offsets + j] = A[offsets + j] + B[offsets + j]

    return main
```

TileLang 将生成以下 CUDA 代码：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    if (((i * 256) + ((int)threadIdx.x)) < 2047) {
      C[((i * 256) + ((int)threadIdx.x))] = (A[((i * 256) + ((int)threadIdx.x))] + B[((i * 256) + ((int)threadIdx.x))]);
    }
  }
}
```

我们可以观察到 TileLang 没有应用诸如向量化或合并内存访问等优化。实际上，除了尾部的数据组外，所有其他线程本可以执行更优化的代码。

## TileLang、CUDA 和 CuTe 的比较

在后续示例中，本教程将使用向量加法运算以保持简洁。

通常，刚接触 CUDA 编程的人经常编写类似这样风格的 CUDA 代码：

```c++
// 向量加法
__global__ void elementwise_add(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}
```

上面的代码为每个线程分配计算一个元素，这显然是低效的，因为没有利用合并内存访问和向量化等常见加速技术。然而，用类似逻辑（如基于循环的遍历）编写的 TileLang 代码可以被编译器优化为高效的实现，这对初学者更友好。此外，编译器最终生成的代码是可观察的，为优化过程提供了透明度。

TileLang 为编译后内核生成的 CUDA 代码可以通过 `kernel.get_kernel_source()` 方法获取。以下是第 1 节向量加法示例生成的 CUDA 代码：

```cu
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  if (((int)threadIdx.x) < 32) {
    uint4 __1;
      uint4 v_ = *(uint4*)(A + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      uint4 v__1 = *(uint4*)(B + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8)));
      ((nv_bfloat162*)(&(__1.x)))->x = (((nv_bfloat162*)(&(v_.x)))->x+((nv_bfloat162*)(&(v__1.x)))->x);
      ((nv_bfloat162*)(&(__1.x)))->y = (((nv_bfloat162*)(&(v_.x)))->y+((nv_bfloat162*)(&(v__1.x)))->y);
      ((nv_bfloat162*)(&(__1.y)))->x = (((nv_bfloat162*)(&(v_.y)))->x+((nv_bfloat162*)(&(v__1.y)))->x);
      ((nv_bfloat162*)(&(__1.y)))->y = (((nv_bfloat162*)(&(v_.y)))->y+((nv_bfloat162*)(&(v__1.y)))->y);
      ((nv_bfloat162*)(&(__1.z)))->x = (((nv_bfloat162*)(&(v_.z)))->x+((nv_bfloat162*)(&(v__1.z)))->x);
      ((nv_bfloat162*)(&(__1.z)))->y = (((nv_bfloat162*)(&(v_.z)))->y+((nv_bfloat162*)(&(v__1.z)))->y);
      ((nv_bfloat162*)(&(__1.w)))->x = (((nv_bfloat162*)(&(v_.w)))->x+((nv_bfloat162*)(&(v__1.w)))->x);
      ((nv_bfloat162*)(&(__1.w)))->y = (((nv_bfloat162*)(&(v_.w)))->y+((nv_bfloat162*)(&(v__1.w)))->y);
    *(uint4*)(C + ((((int)blockIdx.x) * 256) + (((int)threadIdx.x) * 8))) = __1;
  }
}
```

在上面的代码中，TileLang 不仅自动将块级并行映射到线程，还应用了向量化和合并内存访问等优化。

虽然 TileLang 对上述情况进行了各种优化，但其行为有时可能看起来不太直观。例如，当目标是 256 个线程的任务处理时，应用向量化会导致每个线程计算 8 个数据元素——实际上只使用了 32 个活跃线程。有趣的是，内核启动配置仍然保留了原始分配的 256 个线程。

在这种情况下，显式指定每个线程计算的元素数量可以帮助"引导" TileLang 的代码生成过程，产生与预期设计更一致的实现。

```python
def elementwise_add(N, num_per_thread=8, threads=256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as (b_x):
            # 向量加法
            for i, j in T.Parallel(threads, num_per_thread):
                offsets = (b_x * threads + i) * num_per_thread
                C[offsets + j] = A[offsets + j] + B[offsets + j]

    return main
```

上面示例生成的对应 CUDA 代码如下：

```c++
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  uint4 __1;
    uint4 v_ = *(uint4*)(A + (((int)threadIdx.x) * 8));
    uint4 v__1 = *(uint4*)(B + (((int)threadIdx.x) * 8));
    ((nv_bfloat162*)(&(__1.x)))->x = (((nv_bfloat162*)(&(v_.x)))->x+((nv_bfloat162*)(&(v__1.x)))->x);
    ((nv_bfloat162*)(&(__1.x)))->y = (((nv_bfloat162*)(&(v_.x)))->y+((nv_bfloat162*)(&(v__1.x)))->y);
    ((nv_bfloat162*)(&(__1.y)))->x = (((nv_bfloat162*)(&(v_.y)))->x+((nv_bfloat162*)(&(v__1.y)))->x);
    ((nv_bfloat162*)(&(__1.y)))->y = (((nv_bfloat162*)(&(v_.y)))->y+((nv_bfloat162*)(&(v__1.y)))->y);
    ((nv_bfloat162*)(&(__1.z)))->x = (((nv_bfloat162*)(&(v_.z)))->x+((nv_bfloat162*)(&(v__1.z)))->x);
    ((nv_bfloat162*)(&(__1.z)))->y = (((nv_bfloat162*)(&(v_.z)))->y+((nv_bfloat162*)(&(v__1.z)))->y);
    ((nv_bfloat162*)(&(__1.w)))->x = (((nv_bfloat162*)(&(v_.w)))->x+((nv_bfloat162*)(&(v__1.w)))->x);
    ((nv_bfloat162*)(&(__1.w)))->y = (((nv_bfloat162*)(&(v_.w)))->y+((nv_bfloat162*)(&(v__1.w)))->y);
  *(uint4*)(C + (((int)threadIdx.x) * 8)) = __1;
}
```
很好，这段 CUDA 代码与传统编程实践高度对齐，更加熟悉和直观。

但如果我们向 TileLang 提供额外的提示会怎样？例如，使用 `T.copy(...)` 操作显式指定寄存器拷贝。以下示例演示了一种向量加法实现。与之前的示例不同，这段代码在执行计算之前显式地将数据加载到寄存器中。

```python
def elementwise_add(N, NUM_ELE_PER_THREAD=8, threads=256, dtype=T.bfloat16):

    @T.prim_func
    def main(A: T.Tensor((N), dtype), B: T.Tensor((N), dtype), C: T.Tensor((N), dtype)):
        with T.Kernel(T.ceildiv(N, threads * NUM_ELE_PER_THREAD), threads=threads) as (b_x):
            A_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            B_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)
            C_register = T.alloc_fragment((threads * NUM_ELE_PER_THREAD), dtype)

            s_start = b_x * threads * NUM_ELE_PER_THREAD
            s_end = (b_x + 1) * threads * NUM_ELE_PER_THREAD

            # LDG. 128
            T.copy(
                A[s_start:s_end],
                A_register,
            )
            T.copy(
                B[s_start:s_end],
                B_register,
            )

            # 向量加法
            for tid, i in T.Parallel(threads, NUM_ELE_PER_THREAD):
                C_register[tid * NUM_ELE_PER_THREAD + i] = (
                    A_register[tid * NUM_ELE_PER_THREAD + i] +
                    B_register[tid * NUM_ELE_PER_THREAD + i])

            # STG. 128
            T.copy(
                C_register,
                C[s_start:s_end],
            )

    return main
```

在上面的例子中，每个线程负责计算 8 个元素。`T.copy(...)` 方法在块级别运行，TileLang 会自动将数据搬运操作映射到各个线程。这种设计可能对 CUDA 开发者更为直观。现在让我们分析这个实现生成的 CUDA 代码。

```c++
// 编译时 N 设置为 8192 * 8192
extern "C" __global__ void __launch_bounds__(256) main_kernel(bfloat16_t* __restrict__ A, bfloat16_t* __restrict__ B, bfloat16_t* __restrict__ C) {
  bfloat16_t A_register[8];
  bfloat16_t B_register[8];
  *(uint4*)(A_register + 0) = *(uint4*)(A + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  *(uint4*)(B_register + 0) = *(uint4*)(B + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8)));
  uint4 __1;
    uint4 v_ = *(uint4*)(A_register + 0);
    uint4 v__1 = *(uint4*)(B_register + 0);
    ((nv_bfloat162*)(&(__1.x)))->x = (((nv_bfloat162*)(&(v_.x)))->x+((nv_bfloat162*)(&(v__1.x)))->x);
    ((nv_bfloat162*)(&(__1.x)))->y = (((nv_bfloat162*)(&(v_.x)))->y+((nv_bfloat162*)(&(v__1.x)))->y);
    ((nv_bfloat162*)(&(__1.y)))->x = (((nv_bfloat162*)(&(v_.y)))->x+((nv_bfloat162*)(&(v__1.y)))->x);
    ((nv_bfloat162*)(&(__1.y)))->y = (((nv_bfloat162*)(&(v_.y)))->y+((nv_bfloat162*)(&(v__1.y)))->y);
    ((nv_bfloat162*)(&(__1.z)))->x = (((nv_bfloat162*)(&(v_.z)))->x+((nv_bfloat162*)(&(v__1.z)))->x);
    ((nv_bfloat162*)(&(__1.z)))->y = (((nv_bfloat162*)(&(v_.z)))->y+((nv_bfloat162*)(&(v__1.z)))->y);
    ((nv_bfloat162*)(&(__1.w)))->x = (((nv_bfloat162*)(&(v_.w)))->x+((nv_bfloat162*)(&(v__1.w)))->x);
    ((nv_bfloat162*)(&(__1.w)))->y = (((nv_bfloat162*)(&(v_.w)))->y+((nv_bfloat162*)(&(v__1.w)))->y);
  *(uint4*)(A_register + 0) = __1;
  *(uint4*)(C + ((((int)blockIdx.x) * 2048) + (((int)threadIdx.x) * 8))) = *(uint4*)(A_register + 0);
}
```

我们观察到出现了两个额外的寄存器 `A_register` 和 `B_register`。不过在实际计算中，这些寄存器只是简单地被重新赋值给了 `v_` 和 `v__1`。

为了评估复杂度，可以使用 CuTe 实现相同的逐元素加法算子并与 TileLang 版本进行比较。对应的 CuTe 代码如下：

```c++
template<int NUM_ELE_PER_THREAD=8>
__global__ void elementwise_add(nv_bfloat16* C,
                                 const nv_bfloat16* A,
                                 const nv_bfloat16* B,
                                 int N) {
  using namespace cute;

  const int idx = threadIdx.x + blockIdx.x * blockDim.x;

  Tensor t_C = make_tensor(make_gmem_ptr(C), make_shape(N));
  Tensor t_A = make_tensor(make_gmem_ptr(A), make_shape(N));
  Tensor t_B = make_tensor(make_gmem_ptr(B), make_shape(N));

  Tensor t_C_tile = local_tile(t_C, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor t_A_tile = local_tile(t_A, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));
  Tensor t_B_tile = local_tile(t_B, make_shape(Int<NUM_ELE_PER_THREAD>{}), make_coord(idx));

  Tensor reg_buffer_A = make_tensor_like(t_A_tile);
  Tensor reg_buffer_B = make_tensor_like(t_B_tile);
  Tensor reg_buffer_C = make_tensor_like(t_C_tile);

  // LDG. 128
  copy(t_A_tile, reg_buffer_A);
  copy(t_B_tile, reg_buffer_B);

  auto reg_C_vector = recast<nv_bfloat162>(reg_buffer_C);
  auto reg_A_vector = recast<nv_bfloat162>(reg_buffer_A);
  auto reg_B_vector = recast<nv_bfloat162>(reg_buffer_B);

  // 执行向量化加法
#pragma unroll
  for (int vec_idx = 0; vec_idx < size(reg_C_vector); ++vec_idx) {
    reg_C_vector(vec_idx) = reg_A_vector(vec_idx) + reg_B_vector(vec_idx);
  }

  auto reg_C_flat = recast<nv_bfloat16>(reg_C_vector);

  // STG. 128
  copy(reg_C_flat, t_C_tile);
}
```

## 结论

本教程展示了使用 TileLang 实现逐元素加法算子的过程，同时比较了多种设计方案。TileLang 显著降低了 CUDA 编程的复杂性，以最少的代码实现了高性能。尽管如此，使用 TileLang 需要仔细关注特定的实现细节。为确保计算效率，仔细检查生成的 CUDA 内核是至关重要的。

---

**参考文献：**

[1] CuTe 代码实现的灵感来源于本博客讨论的技术：https://zhuanlan.zhihu.com/p/690703999
