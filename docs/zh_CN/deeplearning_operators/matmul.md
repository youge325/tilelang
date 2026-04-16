# 使用 Tile 库的通用矩阵-矩阵乘法

<div style="text-align: left;">
    <em>Author:</em> <a href="https://github.com/LeiWang1999">Lei Wang</a>
</div>

:::{warning}
:class: myclass1 myclass2
:name: a-tip-reference

   本文档仍处于**实验阶段**，内容可能不够完整。
   欢迎提出建议和改进意见——请提交 PR！
:::

TileLang 是一种专门为编写高性能 GPU 内核设计的领域特定语言（DSL, Domain-Specific Language）。它提供了三个主要的抽象层级：

- **第一层级：** 用户只需编写纯计算逻辑，无需了解或关心硬件细节（如 GPU 缓存、分块等）。编译器或运行时会自动完成调度和优化。这一层级的概念类似于 TVM 的设计思想。

- **第二层级：** 用户了解 GPU 架构概念——如共享内存（Shared Memory）、分块（Tiling）和线程块（Thread Block）——但不一定需要下降到最底层的显式线程控制。这种模式类似于 Triton 的编程模型，你可以编写瓦片级（Tile-level）操作，让编译器完成布局推断、流水线化等工作。

- **第三层级：** 用户完全控制线程级原语，可以编写几乎与手写 CUDA/HIP 内核一样显式的代码。这对于需要管理每一个细节（如 PTX 内联汇编、显式线程行为等）的性能专家非常有用。

```{figure} ../../_static/img/overview.png
:width: 50%
:alt: TileLang 编译流程概览
:align: center

图 1：TileLang 编译流程的高层概览。
```

在本教程中，我们通过一个矩阵乘法示例介绍 TileLang 的第二层级。我们将逐步讲解如何分配共享内存、设置线程块、执行并行复制、流水线化计算，以及调用瓦片级 GEMM 内置函数。随后，我们将展示如何在 Python 中编译和运行内核，比较结果并测量性能。

## 为什么需要另一个 GPU DSL？

TileLang 诞生于以下需求：

1. 在高层次表达能力（如 TVM 或 Triton）与必要时控制细节的灵活性之间取得平衡。
2. 支持多种硬件后端（NVIDIA GPU、AMD GPU、CPU 等）的高效代码生成和调度。
3. 通过内置原语（如 `T.Pipelined`、`T.Parallel`、`T.gemm`）简化调度和内存流水线，同时保留专家级调优选项。

TileLang 的第一层级对一般用户来说非常友好——因为它不需要调度或硬件相关知识——但可能需要较长的自动调优时间，并且在处理某些复杂的内核融合模式（如 Flash Attention）时可能不够灵活。第三层级给予你完全的控制权，但需要付出更多努力，类似于编写原生 CUDA/HIP 内核。因此，第二层级为那些希望编写可移植且相对简洁的代码，同时表达重要架构提示的用户提供了一个平衡点。

## 矩阵乘法示例

```{figure} ../../_static/img/MatmulExample.png
:alt: 矩阵乘法示例
:align: center

```

### 基本结构

以下是一个简化的 1024 x 1024 x 1024 矩阵乘法代码片段。它使用了：

- **`T.Kernel(...)`** 初始化线程块配置（网格维度、块大小等）。
- **`T.alloc_shared(...)`** 分配 GPU 共享内存。
- **`T.alloc_fragment(...)`** 分配用于累加的寄存器片段（Register Fragment）。
- **`T.Pipelined(...)`** 表达沿 K 维度的软件流水线。
- **`T.Parallel(...)`** 并行化数据复制循环。
- **`T.gemm(...)`** 执行瓦片级 GEMM 操作（会映射到相应后端，如 NVIDIA GPU 上的 MMA 指令）。

```python
import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_mma_swizzle_layout

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        # Initialize Kernel Context
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

            # Optional layout hints (commented out by default)
            # T.annotate_layout({
            #     A_shared: make_mma_swizzle_layout(A_shared),
            #     B_shared: make_mma_swizzle_layout(B_shared),
            # })

            # Optional: Enabling swizzle-based rasterization
            # T.use_swizzle(panel_size=10, enable=True)

            # Clear local accumulation
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                # Copy tile of A from global to shared memory
                T.copy(A[by * block_M, ko * block_K], A_shared)

                # Parallel copy tile of B from global to shared memory
                for k, j in T.Parallel(block_K, block_N):
                    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]

                # Perform a tile-level GEMM
                T.gemm(A_shared, B_shared, C_local)

            # Copy result from local (register fragment) to global memory
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

# 1. Create the TileLang function
func = matmul(1024, 1024, 1024, 128, 128, 32)

# 2. JIT-compile the kernel for NVIDIA GPU
jit_kernel = tilelang.compile(func, out_idx=[2], target="cuda")

import torch

# 3. Prepare input tensors in PyTorch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)

# 4. Invoke the JIT-compiled kernel
c = jit_kernel(a, b)
ref_c = a @ b

# 5. Validate correctness
torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
print("Kernel output matches PyTorch reference.")

# 6. Inspect generated CUDA code (optional)
cuda_source = jit_kernel.get_kernel_source()
print("Generated CUDA kernel:\n", cuda_source)

# 7. Profile performance
profiler = jit_kernel.get_profiler()
latency = profiler.do_bench()
print(f"Latency: {latency} ms")
```

### 关键概念

1. **内核上下文（Kernel Context）**：

```python
with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
    ...
```

- 根据 N/block_N 和 M/block_M 设置块网格维度。
- `threads=128` 指定每个线程块使用 128 个线程。编译器会推断循环如何映射到这些线程。

```{figure} ../../_static/img/Parallel.png
:alt: 并行示意图
:align: center

```

2. **共享内存与片段内存（Shared & Fragment Memory）**：

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)
```

- `T.alloc_shared` 在整个线程块范围内分配共享内存。
- `T.alloc_fragment` 分配用于局部累加的寄存器空间。虽然写作 `(block_M, block_N)`，但编译器的布局推断会将这块空间的切片分配给各个线程。

3. **软件流水线（Software Pipelining）**：

```python
for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
    ...
```

- `T.Pipelined` 自动安排异步复制和计算指令，使内存操作与算术运算重叠执行。
- 参数 `num_stages=3` 表示流水线深度。

```{figure} ../../_static/img/software_pipeline_inference.png
:alt: 软件流水线推断
:align: center

```

4. **并行复制（Parallel Copy）**：

```python
for k, j in T.Parallel(block_K, block_N):
    B_shared[k, j] = B[ko * block_K + k, bx * block_N + j]
```

- `T.Parallel` 将循环标记为线程级并行化。
- 编译器会将这些循环映射到块中可用的线程。

5. **瓦片级 GEMM（Tile-Level GEMM）**：

```python
T.gemm(A_shared, B_shared, C_local)
```

- 单次调用即可使用指定的缓冲区执行瓦片级矩阵乘法。
- 在底层，对于 NVIDIA 目标，可以使用 CUTLASS/Cute 或 WMMA 指令。在 AMD GPU 上，TileLang 使用单独的 HIP 或可组合内核（Composable Kernel）方法。

6. **结果回写（Copying Back Results）**：

```python
T.copy(C_local, C[by * block_M, bx * block_N])
```

- 计算完成后，局部寄存器片段中的数据被写回全局内存。

## 与其他 DSL 的比较

TileLang 的第二层级在概念上类似于 Triton——用户可以控制分块和并行化，同时让编译器处理许多底层细节。然而，TileLang 还具有以下特点：

- 允许显式的内存布局标注（如 `make_mma_swizzle_layout`）。
- 支持灵活的流水线传递（`T.Pipelined`），可以自动推断或手动定义。
- 支持在单个程序中混合不同层级——例如，你可以将内核的某些部分用第三层级（线程原语）编写以实现细粒度的 PTX/内联汇编控制，其余部分保持在第二层级。

## 不同平台上的性能

```{figure} ../../_static/img/op_benchmark_consistent_gemm_fp16.png
:alt: 不同平台上的性能
:align: center

```

经过适当调优（例如使用自动调优器），TileLang 在各种 GPU 上实现了与厂商库和 Triton 相当甚至更优的性能。在内部基准测试中，对于 FP16 矩阵乘法（如 4090、A100、H100、MI300X），TileLang 的表现如下：

- 在 RTX 4090 上相比 cuBLAS 约有 1.1 倍加速
- 在 A100 上约为 0.97 倍（与 cuBLAS 持平）
- 在 H100 上约为 1.0 倍
- 在 MI300X 上约为 1.04 倍
- 与 Triton 相比，加速比在 1.08 倍到 1.25 倍之间，取决于硬件。

这些测量结果会因分块大小、流水线阶段数和硬件能力的不同而有所变化。

## 总结

本教程演示了一个使用 TileLang 第二层级编写的矩阵乘法内核。仅用几行代码：

1. 我们分配了共享内存和寄存器片段。
2. 我们沿 K 维度对加载和计算进行了流水线化。
3. 我们使用并行复制高效地从全局内存加载数据块。
4. 我们调用 `T.gemm` 来执行瓦片级矩阵乘法。
5. 我们通过 PyTorch 验证了正确性并检查了性能。

通过在高层抽象（如 `T.copy`、`T.Pipelined`、`T.gemm`）与在需要时标注布局或降级到线程原语（第三层级）的能力之间取得平衡，TileLang 既用户友好又高度可调。我们鼓励你尝试不同的分块大小、流水线深度或显式调度，以观察在不同 GPU 上性能如何变化。

如需更高级的用法——包括部分降级（Partial Lowering）、显式控制线程原语或使用内联汇编——可以探索第三层级。同时，如果只需要纯函数式表达和高层调度自动调优，可以考虑第一层级。

## 延伸资源

- [TileLang GitHub](https://github.com/tile-ai/tilelang)
- [BitBLAS](https://github.com/tile-ai/bitblas)
- [Triton](https://github.com/openai/triton)
- [Cutlass](https://github.com/NVIDIA/cutlass)
- [PyCUDA](https://documen.tician.de/pycuda/)  <!-- codespell:ignore -->
