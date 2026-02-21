# 指令集

本页总结了 TileLang DSL 层面可用的核心"指令"，它们如何映射到
硬件概念，以及如何正确使用它们。

## 快速分类
- 数据搬运：`T.copy`、`T.c2d_im2col`，在 Global ↔ Shared ↔ Fragment 之间搬运
- 计算原语：`T.gemm`/`T.gemm_sp`，逐元素数学运算（`T.exp`、`T.max`），
  规约运算（`T.reduce_sum`、`T.cumsum`、warp 规约器）
- 控制辅助：`T.clear`/`T.fill`、`T.reshape`/`T.view`
- 诊断：`T.print`、`T.device_assert`
- 高级：原子操作、内存屏障、warp‑group 操作

## 数据搬运

使用 `T.copy(src, dst, *, coalesced_width=None, disable_tma=False, eviction_policy=None, loop_layout=None)`
在不同内存作用域之间搬运 tile。它接受 `tir.Buffer`、`BufferLoad` 或
`BufferRegion`；范围在可能时自动推断或广播。

```python
# Global → Shared tiles（范围从 dst 推断）
T.copy(A[by * BM, ko * BK], A_s)
T.copy(B[ko * BK, bx * BN], B_s)

# Fragment/Register → Global（写回结果）
T.copy(C_f, C[by * BM, bx * BN])
```

语义
- 范围从参数推断；缺失的维度广播到另一方的秩。
- 访问模式在降级过程中被合法化和合并（coalesced）。在 HL 模式下
  不需要显式向量化。
- 安全性：LegalizeSafeMemoryAccess pass 在访问可能越界时自动插入
  边界守护，在证明安全时则省略。

其他辅助函数
- `T.c2d_im2col(img, col, ...)`：用于卷积风格变换的便捷函数。

## 计算原语

GEMM 和稀疏 GEMM
- `T.gemm(A_shared, B_shared, C_fragment)`：使用共享内存输入和 fragment
  累加器计算 tile GEMM；降级为目标特定的 tensor core。
- `T.gemm_sp(...)`：2:4 稀疏 tensor core 变体（参见示例和 README）。

规约和扫描
- `T.reduce_sum`、`T.reduce_max`、`T.reduce_min`、`T.cumsum`，以及 warp
  规约器（`T.warp_reduce_sum` 等）。
- 通过 `T.alloc_fragment` + `T.clear` 或 `T.fill` 分配和初始化累加器。

逐元素数学运算
- 大多数数学操作与 TVM TIR 对应：`T.exp`、`T.log`、`T.max`、`T.min`、`T.rsqrt`、
  `T.sigmoid` 等。可在循环内自由组合。

Reshape/view（无拷贝）
- `T.reshape(buf, new_shape)` 和 `T.view(buf, shape=None, dtype=None)` 创建
  共享存储的新视图，并强制执行 shape/dtype 检查。

## 同步（HL 用法）

在 HL 流水线中，通常不需要编写显式屏障。PipelinePlanning/
InjectSoftwarePipeline/InjectTmaBarrier 等 pass 会在后台协调
生产者/消费者顺序和线程同步。

如果需要调试或显式检查：
- `T.device_assert(cond, msg='')` 在 CUDA 目标上发出设备端断言。
- `T.print(obj, msg='...')` 从单个线程安全地打印标量或缓冲区。

## 综合示例：GEMM Tile

```python
@T.prim_func
def gemm(
    A: T.Tensor((M, K), 'float16'),
    B: T.Tensor((K, N), 'float16'),
    C: T.Tensor((M, N), 'float16'),
):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        A_s = T.alloc_shared((BM, BK), 'float16')
        B_s = T.alloc_shared((BK, BN), 'float16')
        C_f = T.alloc_fragment((BM, BN), 'float32')
        T.clear(C_f)

        for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
            T.copy(A[by * BM, ko * BK], A_s)  # Global → Shared
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)             # 计算到 fragment

        T.copy(C_f, C[by * BM, bx * BN])      # 写回
```

## 指令参考（简明版）

以下是按类别分组的 TileLang 指令简明列表。完整的函数签名、
行为、约束和示例请参考 API 参考文档（`autoapi/tilelang/index`）。

数据搬运
- `T.copy(src, dst, ...)`：在 Global/Shared/Fragment 之间搬运 tile。
- `T.c2d_im2col(img, col, ...)`：用于卷积的 2D im2col 变换。

内存分配和描述符
- `T.alloc_shared(shape, dtype, scope='shared.dyn')`：分配共享内存缓冲区。
- `T.alloc_fragment(shape, dtype, scope='local.fragment')`：分配 fragment。
- `T.alloc_var(dtype, [init], scope='local.var')`：标量变量缓冲区（1 个元素）。
- `T.alloc_barrier(arrive_count)`：分配并初始化一个或多个 mbarrier。
- `T.alloc_tmem(shape, dtype)`：张量内存（TMEM）缓冲区（Hopper+）。
- `T.alloc_reducer(shape, dtype, op='sum', replication=None)`：规约器缓冲区。
- `T.alloc_descriptor(kind, dtype)`：通用描述符分配器。
  - `T.alloc_wgmma_desc(dtype='uint64')`
  - `T.alloc_tcgen05_smem_desc(dtype='uint64')`
  - `T.alloc_tcgen05_instr_desc(dtype='uint32')`
- `T.empty(shape, dtype='float32')`：声明函数输出张量。

计算原语
- `T.gemm(A_s, B_s, C_f)`：Tile GEMM 到 fragment 累加器。
- `T.gemm_sp(...)`：稀疏（2:4）tensor core GEMM。
- 规约：`T.reduce_sum/max/min/abssum/absmax`，位运算 `and/or/xor`。
- 扫描：`T.cumsum`，完成：`T.finalize_reducer`。
- Warp 规约器：`T.warp_reduce_sum/max/min/bitand/bitor`。
- 逐元素数学：TIR 操作（`T.exp`、`T.log`、`T.max`、`T.min`、`T.rsqrt` 等）。
- 快速数学：`T.__log/__log2/__log10/__exp/__exp2/__exp10/__sin/__cos/__tan`。
- IEEE 数学：`T.ieee_add/sub/mul/fmaf`（可配置舍入方式）。
- 辅助函数：`T.clear(buf)`、`T.fill(buf, value)`。
- 视图：`T.reshape(buf, shape)`、`T.view(buf, shape=None, dtype=None)`。

诊断
- `T.print(obj, msg='')`：从单个线程打印标量/缓冲区。
- `T.device_assert(cond, msg='')`：设备端断言（CUDA）。

逻辑辅助
- `T.any_of(a, b, ...)`、`T.all_of(a, b, ...)`：多项谓词。

标注辅助
- `T.use_swizzle(panel_size=..., enable=True)`：光栅化提示。
- `T.annotate_layout({...})`：为缓冲区附加显式布局。
- `T.annotate_safe_value(var, ...)`：安全性/常量提示。
- `T.annotate_l2_hit_ratio(buf, ratio)`：缓存行为提示。

同步辅助
- `T.pdl_trigger()`：为当前内核发出程序化启动完成信号。
- `T.pdl_sync()`：等待内核依赖关系满足。

原子操作
- `T.atomic_add(dst, value, memory_order=None, return_prev=False, use_tma=False)`。
- `T.atomic_addx2(dst, value, return_prev=False)`；`T.atomic_addx4(...)`。
- `T.atomic_max(dst, value, memory_order=None, return_prev=False)`。
- `T.atomic_min(dst, value, memory_order=None, return_prev=False)`。
- `T.atomic_load(dst)`、`T.atomic_store(dst, value)`。

自定义内置函数
- `T.dp4a(A, B, C)`：4 元素点积累加。
- `T.clamp(x, lo, hi)`：钳位到 [lo, hi]。
- `T.loop_break()`：通过内置函数跳出当前循环。

屏障、TMA、warp‑group
- 屏障：`T.alloc_barrier(arrive_count)`。
- 奇偶操作：`T.mbarrier_wait_parity(barrier, parity)`、`T.mbarrier_arrive(barrier)`。
- Expect tx：`T.mbarrier_expect_tx(...)`；语法糖：`T.barrier_wait(id, parity=None)`。
- TMA：`T.create_tma_descriptor(...)`、`T.tma_load(...)`、
  `T.tma_store_arrive(...)`、`T.tma_store_wait(...)`。
- 代理/围栏：`T.fence_proxy_async(...)`、`T.warpgroup_fence_operand(...)`。
- Warp‑group：`T.warpgroup_arrive()`、`T.warpgroup_commit_batch()`、
  `T.warpgroup_wait(num_mma)`、`T.wait_wgmma(id)`。

Lane/Warp 索引
- `T.get_lane_idx(warp_size=None)`：warp 内的 lane id。
- `T.get_warp_idx_sync(warp_size=None)`：规范 warp id（同步）。
- `T.get_warp_idx(warp_size=None)`：规范 warp id（无同步）。
- `T.get_warp_group_idx(warp_size=None, warps_per_group=None)`：组 id。

寄存器控制
- `T.set_max_nreg(reg_count, is_inc)`、`T.inc_max_nreg(n)`、`T.dec_max_nreg(n)`。
- `T.annotate_producer_reg_dealloc(n=24)`、`T.annotate_consumer_reg_alloc(n=240)`。
- `T.no_set_max_nreg()`、`T.disable_warp_group_reg_alloc()`。

## 关于数据类型的说明

数据类型接受三种等效形式：
- 字符串：`'float32'`
- TileLang dtype：`T.float32`
- 框架 dtype：`torch.float32`
所有形式在内部都会被标准化。详情请参阅类型系统。
