# 使用 Tile 库的稀疏矩阵-矩阵乘法

<div style="text-align: left;">
    <em>Author:</em> <a href="https://github.com/botbw">botbw</a>
</div>

:::{warning}
   本文档仍处于**实验性**阶段，内容可能不完整。

   此功能仍处于**实验性**阶段，需要进一步优化。

   非常欢迎提出建议和改进——请提交 PR！
:::

:::{tip}
建议先阅读 `docs/deeplearning_operators/matmul.md`。

示例代码可在 `examples/gemm_sp` 中找到。
:::

## NVIDIA Ampere 架构中的结构化稀疏

自 Ampere 架构（sm80 及以上）起，稀疏性（Sparsity）支持已被集成到 Tensor Core 中。这允许将 2:4（或对于 32 位数据类型为 1:2）半结构化矩阵压缩为其非零值及相关元数据（Metadata），然后输入到 Tensor Core 中。与等效的稠密计算相比，这可以实现高达 **2 倍的吞吐量**。

:::{warning}
   本教程主要侧重于 CUDA，因为 ROCm 尚不支持此功能。但是，AMD 在 MI300X 等 GPU 的矩阵核心中提供了类似的能力。
:::

```{figure} ../../_static/img/sparse_mma_storage_example.png
:align: center

图：稀疏 MMA 存储示例（来自 PTX 文档）
```

## 压缩稠密张量

要利用稀疏 Tensor Core，必须先将稠密张量（Dense Tensor）**压缩**为其非零值及对应的元数据。

`PyTorch` 和 `vLLM` 都使用 `CUTLASS` 作为其计算后端（参见参考资料 [这里](https://github.com/pytorch/pytorch/blob/a8d6afb511a69687bbb2b7e88a3cf67917e1697e/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredOps.cu#L47) 和 [这里](https://github.com/vllm-project/vllm/blob/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/csrc/sparse/cutlass/sparse_scaled_mm_c3x.cuh#L116)），利用 `CUTLASS` 内置的压缩器（Compressor）（或在 `PyTorch` 中重新实现它）。

在 `tilelang.utils.sparse` 中提供了一组**兼容 CUTLASS** 的压缩器，可以传入稠密张量以及其他必要参数（例如 sm90 的 block_K、转置选项）来执行压缩。

```python
from tilelang.utils.sparse import compress
A_sparse, E = compress(A, transposed=trans_A, block_k=block_K)
```

其中，`A_sparse` 包含 `A` 的所有非零元素，而 `E` 存储了重建原始稀疏模式所需的对应元数据（索引信息）。

> 注意：使用 CUTLASS 压缩器时，`A_sparse`/`A` 中的位置与 `E` 之间不存在简单的位置对应关系。（即如果将元数据视为 int4 张量，[n, k] 处的 4 个元素组不匹配 [n, k] 处的 4 位元数据）
元数据在内部被重新排列以优化内存访问模式（例如用于 ldsm 指令和向量化加载）。
更多信息请参见**关于 `gemm_sp` 和 `gemm_sp_v2` 的说明**。

## 使用 CUTLASS 压缩器的 `T.gemm_sp`

:::{warning}

强烈建议使用 T.gemm_sp_v2，因为它具有更大的灵活性和更快的编译速度。

:::

2:4 稀疏 GEMM 内核与其稠密版本类似，不同之处在于它还需要处理相关的元数据。

请查看下方内核代码中的注释，了解所需的修改。

```python
def matmul_sp_sm80(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    trans_A,
    trans_B,
):
    is_8_bit = "8" in in_dtype
    metadata_dtype = 'int32' if is_8_bit else 'int16'
    E_factor = SparseTensorCoreIntrinEmitter.E_FACTOR_MAP[in_dtype][metadata_dtype]  # Calculate shape for given datatypes
    A_sparse_shape = (M, K // 2) if not trans_A else (K // 2, M)
    B_shape = (K, N) if not trans_B else (N, K)
    A_shared_shape = (block_M, block_K // 2) if not trans_A else (block_K // 2, block_M)
    B_shared_shape = (block_K, block_N) if not trans_B else (block_N, block_K)

    import tilelang.language as T

    @T.prim_func
    def main(
            A_sparse: T.Tensor(A_sparse_shape, in_dtype),
            E: T.Tensor((M, K // E_factor), metadata_dtype),
            B: T.Tensor(B_shape, in_dtype),
            C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // E_factor), metadata_dtype)  # Allocate smem for metadata
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.annotate_layout({  # Annotate reordered cutlass metadata layout
                E:
                    make_cutlass_metadata_layout(E, mma_dtype=in_dtype, arch="8.0"),
                E_shared:
                    make_cutlass_metadata_layout(
                        E_shared, mma_dtype=in_dtype, arch="8.0"),
            })
            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(E[by * block_M, k * block_K // E_factor], E_shared)
                if trans_A:
                    T.copy(A_sparse[k * block_K // 2, by * block_M], A_shared)
                else:
                    T.copy(A_sparse[by * block_M, k * block_K // 2], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, trans_A, trans_B)  # Call gemm_sp with non-zero values and metadata
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main
```

在底层，`gemm_sp` 调用从 `CUTLASS` 改编而来的模板，并且必须使用 `T.annotate_layout` 指定兼容的元数据布局。

## 使用自定义压缩器的 `T.gemm_sp_v2`

要迁移到 `gemm_sp_v2`，只需替换 `gemm_sp` 的出现即可。

与 `gemm_sp` 不同，`gemm_sp_v2` 可以在不使用 `T.annotate_layout` 的情况下运行，并且还支持用户自定义的布局和压缩器。

元数据存储在 `(u)int8`/`(u)int16`/`(u)int32` 张量中，其中**每 4 位块表示四个连续元素中非零元素的两个 2 位索引**。这里，我们从一个 `int16` 的示例开始，这是 Ampere GPU 上 `bf16` 和 `fp16` 的**默认数据类型**。

假设我们有以下行向量：
```python
t = tensor([[0, 7, 0, 3], [1, 5, 0, 0], [0, 0, 2, 4], [9, 0, 9, 0]], dtype=torch.float16).flatten()
```

非零元素及其对应的索引为：

```python
t_sp = tensor([[7, 3], [1, 5], [2, 4], [9, 9]], dtype=torch.float16).flatten()
indices = tensor([[1, 3], [0, 1], [2, 3], [0, 2]], dtype=torch.float16).flatten()
```

对应的 uint16 元数据为：
```python
# metadata_bits = tensor([0b1101, 0b0100, 0b1110, 0b1000])
# Note: storage uses little-endian order: tensor(0b1000111001001101, dtype=torch.int16)
# Note: the above code is not runnable in python as the interpreter won't take the binary
#       as 2's complement
metadata_int16 = tensor(-29107)
```

你可以使用以下工具函数解码 int16 元数据张量：
```python
def decode_metadata(meta: torch.Tensor) -> torch.Tensor:
    assert meta.dtype is torch.int16
    groups_per_meta = 16 // 4
    out = []
    for g in range(groups_per_meta):
        group_bits = (meta >> (g * 4)) & 0xF
        idx0 = group_bits & 0x3
        idx1 = (group_bits >> 2) & 0x3
        out.append(torch.stack([idx0, idx1], dim=-1))
    return torch.concat(out, dim=-1).view(meta.shape[0], -1)
```

压缩器可以在 `PyTorch`/`NumPy` 层面或内核层面实现。

例如，`PyTorch` 提供了一个 Ampere 压缩器 [在这里](https://github.com/pytorch/pytorch/blob/267d0197bfca0232488d51dd1ff735d619adc2cf/torch/sparse/_semi_structured_conversions.py#L47-L179)。请注意，在此实现中，应用了一个 [置换（Permutation）](https://github.com/pytorch/pytorch/blob/267d0197bfca0232488d51dd1ff735d619adc2cf/torch/sparse/_semi_structured_conversions.py#L173-L175) 以匹配 CUTLASS 的元数据布局。如果在使用 `gemm_sp_v2` 时不标注元数据布局，你的压缩器应该与 PyTorch 示例的行为一致——但无需使用 `_calculate_meta_reordering_scatter_offsets` 函数。

如果你想在内核中使用自定义元数据布局，一种方法是在 `TileLang` 中定义布局，然后将相同的布局应用于压缩器内核和 matmul_sp 内核。

```python

@tilelang.jit(out_idx=[1, 2], pass_configs={
    tilelang.PassConfigKey.TIR_DISABLE_VECTORIZE: True,
})
def compress_kernel(M, K, block_M, block_K, dtype, use_cutlass_layout):
    e_factor, e_dtype = ARCH_INFO["8.0"]
    e_K = K // e_factor
    elem, group = 2, 4

    assert M % block_M == 0, "M must be divisible by block_M"
    assert K % block_K == 0, "K must be divisible by block_K"
    assert K % e_factor == 0, "K must be divisible by e_factor"
    assert block_K % e_factor == 0, "block_K must be divisible by e_factor"

    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), dtype),
        A_sp: T.Tensor((M, K // 2), dtype),
        E: T.Tensor((M, e_K), e_dtype),
    ):
        with T.Kernel(T.ceildiv(M, block_M), T.ceildiv(K, block_K), threads=block_M) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            A_sp_shared = T.alloc_shared((block_M, block_K // 2), dtype)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), e_dtype)
            if use_cutlass_layout:  # NOTE: Make sure compressor metadata layout
                T.annotate_layout({ # is same with your computation kernel
                    E:
                        make_cutlass_metadata_layout(
                            E, mma_dtype="float16", arch="8.0", block_k=block_K),
                    E_shared:
                        make_cutlass_metadata_layout(
                            E_shared,
                            mma_dtype="float16",
                            arch="8.0",
                            block_k=block_K),
                })
            T.clear(A_sp_shared)
            T.clear(E_shared)
            non_zero_cnt = T.alloc_local((1, ), dtype="uint8")
            non_zero_elt_log_idx = T.alloc_local((elem, ), dtype="uint8")
            T.copy(A[bx * block_M, by * block_K], A_shared)
            for tm in T.Parallel(block_M):
                for g_i in range(0, block_K // group):
                    a_k = g_i * group
                    T.clear(non_zero_cnt)
                    T.clear(non_zero_elt_log_idx)
                    for i in range(group):
                        val = A_shared[tm, a_k + i]
                        if val != 0.0:
                            non_zero_elt_log_idx[non_zero_cnt[0]] = i
                            A_sp_shared[tm, a_k // 2 + non_zero_cnt[0]] = val
                            non_zero_cnt[0] += 1
                    if non_zero_cnt[0] == 1 and non_zero_elt_log_idx[0] == 3:
                        non_zero_elt_log_idx[0] = 0
                        non_zero_elt_log_idx[1] = 3
                        A_sp_shared[tm, a_k // 2 + 1] = A_sp_shared[tm, a_k // 2]
                        A_sp_shared[tm, a_k // 2] = 0.0
                    elif non_zero_cnt[0] == 1:
                        A_sp_shared[tm, a_k // 2 + 1] = 0
                        non_zero_elt_log_idx[1] = 3
                    for i in T.serial(elem):
                        val = non_zero_elt_log_idx[i]
                        E_shared[tm, a_k // e_factor] |= T.shift_left(val, 4 * (g_i % (e_factor // group)) + 2 * i)
            T.copy(A_sp_shared, A_sp[bx * block_M, by * block_K // 2])
            T.copy(E_shared, E[bx * block_M, by * block_K // e_factor])

    return kernel
```

## 关于 `gemm_sp` 和 `gemm_sp_v2` 的说明

最初，`T.gemm_sp` 遵循与 `T.gemm` 相同的设计，降低（Lower）为 `CUTLASS` 模板。这本质上要求元数据按照预定的布局在离线阶段进行重排。

然而，固定特定布局会引入几个潜在问题：

1. 调试体验困难：由于重排的索引（包括置换和 Swizzling），调试失败的内核变得很困难。

2. 灵活性有限：例如，将两个压缩张量 `A_sparse_0` 和 `A_sparse_1` 拼接为新的 `A_sparse` 是合理的。然而，拼接它们的元数据 `E_0` 和 `E_1` 可能无效，除非布局在数学上允许这样做。

3. 对齐要求：`CUTLASS` 强制执行严格的对齐检查，许多超参数配置可能导致编译错误。（参考：sm8x 在 `CUTLASS 2` 中实现。）

`T.gemm_sp_v2` 旨在解决这些限制，遵循 `T.gemm_v2` 的方法。它直接降低为 PTX，无需固定的元数据布局。
