# 自动调优

TileLang 内置了一个自动调优器（autotuner），可搜索配置空间
以找到最佳性能的内核，并行编译候选方案，验证正确性，
进行基准测试，并缓存最优结果以便复用。

本指南涵盖两种工作流：
- 装饰器方式：在 `@tilelang.jit` 上面叠加 `@tilelang.autotune(configs=...)`
- 编程方式：`AutoTuner.from_kernel(...).set_*().run()`

还将介绍输入张量的提供方式、验证、缓存以及影响并行度和
缓存行为的环境变量。

## 1) 装饰器方式自动调优

在 `@tilelang.jit` 上方使用 `@tilelang.autotune`，并将可调参数
暴露为带默认值的函数参数。自动调优器会用配置空间中的值
覆盖这些参数。

```python
import tilelang
import tilelang.language as T

def matmul_configs(M, N, K):
    # 示例空间 — 根据目标硬件调整
    tiles = [64, 128]
    stages = [2, 3]
    threads = [128, 256]
    return [
        dict(block_M=BM, block_N=BN, block_K=BK, num_stages=S, threads=TH)
        for BM in tiles
        for BN in tiles
        for BK in [32, 64]
        for S in stages
        for TH in threads
    ]

@tilelang.autotune(configs=matmul_configs, warmup=25, rep=100, timeout=60)
@tilelang.jit(out_idx=[-1])
def matmul(M: int, N: int, K: int,
           block_M: int = 128, block_N: int = 128, block_K: int = 32,
           threads: int = 128, num_stages: int = 3,
           dtype: str = 'float16', accum_dtype: str = 'float32'):

    @T.prim_func
    def kernel(A: T.Tensor((M, K), dtype),
               B: T.Tensor((K, N), dtype),
               C: T.Tensor((M, N), dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_s = T.alloc_shared((block_M, block_K), dtype)
            B_s = T.alloc_shared((block_K, block_N), dtype)
            C_f = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_f)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, ko * block_K], A_s)
                T.copy(B[ko * block_K, bx * block_N], B_s)
                T.gemm(A_s, B_s, C_f)

            T.copy(C_f, C[by * block_M, bx * block_N])

    return kernel

# 用法
# 通过上下文管理器提供输入（推荐，以确保跨配置的可复现性）
import torch
M = N = K = 1024
A = torch.randn(M, K, device='cuda', dtype=torch.float16)
B = torch.randn(K, N, device='cuda', dtype=torch.float16)
C = torch.empty(M, N, device='cuda', dtype=torch.float16)

from tilelang.autotuner import set_autotune_inputs
with set_autotune_inputs(A, B, C):
    tuned_kernel = matmul(M, N, K)   # 编译、调优、返回最佳内核
    tuned_kernel(A, B, C)            # 运行最佳内核
```

注意事项
- `configs` 可以是字典列表或可调用对象 `(args...) -> list[dict]`。每个
  字典的键必须与可调函数参数匹配（如 `block_M`）。
- 装饰器返回一个可调用对象，对每个参数元组运行一次自动调优，
  并在进程内缓存最佳内核。
- 要在调优过程中显式控制输入，请用 `set_autotune_inputs(...)` 包装调用。
  否则使用 `supply_type`（见下文）。

## 2) 编程方式自动调优

使用 `AutoTuner` 类以更显式的方式管理配置和参数。

```python
from tilelang.autotuner import AutoTuner

kernel_factory = matmul  # 上面的函数（已添加 @tilelang.jit）
tuner = AutoTuner.from_kernel(kernel_factory(M, N, K), configs=matmul_configs(M, N, K))

tuner.set_profile_args(
    warmup=25, rep=100, timeout=60,
    supply_type=tilelang.TensorSupplyType.Auto,  # 或提供 supply_prog/ref_prog
    ref_prog=lambda A, B, C: torch.allclose(C, (A @ B).to(C.dtype), rtol=1e-2, atol=1e-2),
)

tuner.set_compile_args(
    target='auto',                  # 或 'cuda'/'hip'/'metal'
    execution_backend='auto',       # 按目标自动解析
    out_idx=[-1],                   # 多输出时返回哪些
    pass_configs={                  # 可选的 TVM pass/标志
        # tilelang.PassConfigKey.EXAMPLE_KEY: value,
    },
)

artifact = tuner.run()             # 编译 + 运行 + 验证所有配置
best_kernel = artifact.kernel      # JITKernel
best_latency = artifact.latency
best_config = artifact.config

# 复用最佳内核
best_kernel(A, B, C)
```

### 示例集（在仓库中）
- examples/gdn/example_chunk_delta_h.py:101 — 使用 `@autotune` 遍历配置
- examples/deepseek_nsa/benchmark/benchmark_nsa_fwd.py:451 — 使用 `@tilelang.autotune`
- examples/quickstart.py:84 — 使用 `get_profiler` 对调优后内核进行性能分析
- examples/hadamard_transform/example_hadamard.py:152 — 自定义 warmup 的性能分析器
- examples/dynamic_shape/example_dynamic.py:94 — 动态形状的性能分析器
- examples/gemm/example_gemm_persistent.py:135 — 持久化 vs 非持久化对比

点击任意路径即可打开代码并比较模式。

## 输入张量的提供方式

调优器需要输入来编译和基准测试内核。通过以下三种方式
提供（按优先级排列）：

1) 上下文管理器（跨配置固定输入）
```python
with set_autotune_inputs(A, B, C):
    tuned = matmul(M, N, K)
```

2) 自定义提供程序
```python
def supply_prog(signature):
    # signature 包含描述 shape/dtype 的 KernelParam 对象
    # 返回与内核参数匹配的 torch 张量列表
    return [A, B, C]

tuner.set_profile_args(supply_prog=supply_prog)
```

3) 通过 `supply_type` 使用内置生成器
- `TensorSupplyType.Auto`（默认）：按 dtype 启发式生成（均匀整数 / 浮点范围）
- `Integer`、`Uniform`、`Normal`、`Randn`、`Zero`、`One`

重要提示
- 内置生成器需要静态形状；如果您的 PrimFunc 使用符号维度（T.dyn），
  请通过方式 (1) 或 (2) 提供具体输入。
- Float8 数据类型需要 PyTorch 2.1+ 以支持 `torch.float8_*`。

## 正确性检查和容差

使用以下验证方法之一：
- `ref_prog`：提供一个参考程序，接收相同输入并检查结果。
  可以返回布尔值或在不匹配时抛出异常。
- `manual_check_prog`：一个可调用对象，检查输出并在不匹配时抛出异常。
- `skip_check=True`：跳过正确性检查（更快，但请谨慎使用）。

通过以下参数控制数值漂移：
- `rtol` 和 `atol`（默认 1e‑2）
- `max_mismatched_ratio`（默认 1%）

## 配置空间和最佳实践

调优内容
- Tile 大小：`block_M`、`block_N`、`block_K`
- 软件流水线：`num_stages`
- 每个块的线程数：`threads`（或 (x, y) 元组）
- 可选：dtype 变体、epilogue、小型调度旋钮

建议
- 从一个可工作的基线开始。先调优一个小而有意义的空间。
- 遵守硬件限制（共享内存字节数、每线程/块的寄存器数、
  每块最大线程数）。预先排除不可能的配置。
- 在相关时保持块大小为向量宽度和 warp 大小的倍数。
- 使用 `set_autotune_inputs` 确保每个配置在相同数据上测量。
- 记录最佳配置，稳定后将其作为默认值固化。

## 并行编译/基准测试和超时

调优器使用线程池并行编译配置，并为每个配置设置超时进行基准测试。
在 CUDA 上，每个工作线程设置当前设备以避免上下文问题。

注意事项
- `timeout` 使用 POSIX 信号；在非 Unix 系统上可能不生效。
- 日志写入工作目录下的 `autotuner.log`。

## 缓存

自动调优器将最佳产物缓存在内存中（每进程）和磁盘上的
`$TILELANG_CACHE_DIR/autotuner` 目录下。缓存键包括：
- TileLang 版本、函数源码、闭包自由变量
- 配置列表、编译参数、性能分析参数

磁盘缓存内容（每个键）
- 最佳配置和延迟：`best_config.json`、`latency.json`
- 内核源码和库：`device_kernel.cu`、`host_kernel.cu`、`kernel_lib.so`（或 `kernel.cubin`/`executable.so`，取决于后端）
- 函数和参数：`function.pkl`、`params.pkl`

通过环境变量控制（tilelang.env）
- `TILELANG_CACHE_DIR`（默认 `~/.tilelang/cache`）
- `TILELANG_TMP_DIR`（默认 `$TILELANG_CACHE_DIR/tmp`）
- 禁用所有内核缓存：`TILELANG_DISABLE_CACHE=1`
- 仅禁用自动调优磁盘缓存：`TILELANG_AUTO_TUNING_DISABLE_CACHE=1`

CPU 工作线程控制
- `TILELANG_AUTO_TUNING_CPU_UTILITIES`（比例，默认 0.9）
- `TILELANG_AUTO_TUNING_CPU_COUNTS`（整数，`-1` 自动）
- `TILELANG_AUTO_TUNING_MAX_CPU_COUNT`（整数，`-1` 无限制）

后端注意事项
- NVRTC 后端持久化 `.cubin` 和 Python 启动器。
- Torch/DLPack 后端可能不会将产物保存到磁盘；在这种情况下，
  仅应用内存缓存并记录警告。

## 替代方案：使用 par_compile 手动遍历

如果您偏好手动控制，可使用 `JITImpl.par_compile` 批量编译配置
并自行驱动基准测试：

```python
@tilelang.jit
def factory(M, N, K, block_M=128, block_N=128, block_K=32):
    @T.prim_func
    def k(A: T.Tensor((M, K), 'float16'),
           B: T.Tensor((K, N), 'float16'),
           C: T.Tensor((M, N), 'float16')):
        ...
    return k

impl = factory  # JITImpl
cfgs = [
    dict(block_M=64, block_N=128, block_K=32),
    dict(block_M=128, block_N=128, block_K=64),
]
kernels = impl.par_compile(cfgs, num_workers=4)
# 现在自行基准测试 kernels[i](A, B, C)
```

## 记录和复用最佳配置

编程方式返回一个 `AutotuneResult`，可以保存并在以后重新加载。
这对 CI、多主机工作流或发布调优配置很有用。

```python
artifact = tuner.run()  # AutotuneResult

# 保存到磁盘
from pathlib import Path
save_dir = Path('out/best/matmul_1024')
artifact.save_to_disk(save_dir, verbose=True)

# 稍后重新加载
from tilelang.autotuner.param import AutotuneResult, CompileArgs
restored = AutotuneResult.load_from_disk(save_dir, CompileArgs())
best = restored.kernel
best(A, B, C)
```

注意事项
- DLPack/Torch 执行后端可能不会持久化编译后的二进制文件；
  在这种情况下，加载时需要重新编译或使用其他后端。
- 目录包含人类可读的 JSON（最佳配置/延迟）和源码。

## 高级：配置空间可调用对象

根据问题规模推导配置空间，以保持搜索针对性和合法性：

```python
def matmul_configs(M, N, K):
    large = min(M, N, K) >= 1024
    tiles = [128] if large else [64, 128]
    for BM in tiles:
        for BN in tiles:
            for BK in [32, 64]:
                for S in [2, 3]:
                    for TH in [128, 256]:
                        yield dict(block_M=BM, block_N=BN, block_K=BK,
                                    num_stages=S, threads=TH)
```

## 设备和后端选择

显式调优编译时选项：
- `target='auto'|'cuda'|'hip'|'metal'`（标准化为 TVM Target）
- `execution_backend='auto'|'tvm_ffi'|'cython'|'nvrtc'|'torch'`
- `pass_configs={...}` 用于切换 TileLang/TVM pass 进行实验

在带有多 GPU 的 CUDA 上，调优器为每个工作线程设置当前设备
以避免上下文混乱。

## 故障排除
- "No configurations to tune"：确保 `configs` 是非空列表或可调用对象。
- 超时：增加 `timeout`；确保输入适合设备内存；验证
  您的参考检查不是瓶颈。
- 动态形状：通过 `set_autotune_inputs` 或自定义 `supply_prog` 提供具体输入。
- 磁盘缓存被禁用：检查 `TILELANG_AUTO_TUNING_DISABLE_CACHE` 和后端。
