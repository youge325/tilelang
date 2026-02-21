自动调优技术实现性能优化
===================================================
<div style="text-align: left;">
<em>作者：</em> <a href="https://github.com/yyttt6">yyttt6</a>
</div>

## 概述

对 Tile Language 程序进行自动调优包含三个主要步骤：

1. 使用 Tile Language 实现目标程序，并保留优化参数
2. 通过手动搜索或 [使用 Carver 自动生成候选配置](#using-carver-to-auto-generate-candidate-configurations) 提供候选配置
3. 并行编译和基准测试候选配置以确定最佳性能

## 矩阵乘法示例

以下示例演示了矩阵乘法的自动调优。代码已简化以提高可读性——完整实现请参阅 `examples/gemm/example_gemm.py`。

### 步骤 1：使用保留参数实现
用户可以使用 Tile Language 实现矩阵乘法，同时保留用于优化的参数：
```python
# 保留用于优化的参数
def kernel(
    block_M=None,
    block_N=None,
    block_K=None,
    num_stages=None,
    thread_num=None,
    enable_rasteration=None,
):
    dtype = "float16"
    accum_dtype = "float"

    # 矩阵乘法实现
    @T.prim_func
    def main(
            A: T.Buffer((M, K), dtype),
            B: T.Buffer((N, K), dtype),
            C: T.Buffer((M, N), dtype),
    ):
        # ...existing code...

    return main
```
### 步骤 2：生成候选配置
手动定义配置或使用组合生成：
```python
configs = [
    {
        "block_M": 128,
        "block_N": 128,
        "block_K": 128,
        "num_stages": 3,
        "thread_num": 128,
        "enable_rasteration": True
    },
        {
        "block_M": 32,
        "block_N": 32,
        "block_K": 32,
        "num_stages": 0,
        "thread_num": 32,
        "enable_rasteration": False
    },
    # ...其他配置...
]
```
也可以通过不同参数的组合遍历来生成
```python
import itertools

block_M = [64, 128, 256]
block_N = [64, 128, 256]
block_K = [32, 64]
num_stages = [0, 1, 2, 3]
thread_num = [128, 256]
enable_rasterization = [True, False]
_configs = list(
    itertools.product(
        block_M,
        block_N,
        block_K,
        num_stages,
        thread_num,
        enable_rasterization,
    ))

configs = [
    {
        "block_M": c[0],
        "block_N": c[1],
        "block_K": c[2],
        "num_stages": c[3],
        "thread_num": c[4],
        "enable_rasteration": c[5]
    } for c in _configs
]
```
### 步骤 3：编译和基准测试
配置 JIT 编译和基准测试设置：
```python
autotuner = AutoTuner.from_kernel(
    kernel=kernel, configs=get_configs(M, N, K, with_roller)).set_compile_args(
        out_idx=[-1],
        supply_type=tl.TensorSupplyType.Integer,
        ref_prog=ref_program,
        skip_check=False,
        target="auto",
    )
result = autotuner.run(warmup=3, rep=20)
out_c = result.kernel(a, b)
```
result 对象包含优化后的内核实现，用户可以直接使用

## 使用 Carver 自动生成候选配置

Carver 是一个轻量级框架，用于生成和排序 tile 配置（也称为 tiling 策略、分块方案或调度提示），适用于常见的 GPU、CPU 和加速器后端。它帮助您探索循环的高效映射，适用于矩阵乘法、逐元素变换等操作以及其他基于归约的内核。

对于常见算子，Carver 提供了预构建的模板（例如 `MatmulTemplate`）：

```python
# 配置 Matmul 模板
arch = CUDA("cuda")
carve_template = MatmulTemplate(
    M=M,
    N=N,
    K=K,
    in_dtype="float16",
    out_dtype="float16",
    accum_dtype="float",
).with_arch(arch)

# 生成 top-k 优化提示（推荐 topk=10）
roller_hints = carve_template.recommend_hints(topk=10)

# 配置候选参数
for hint in roller_hints:

    # ...existing code...

    config["block_M"] = block_m
    config["block_N"] = block_n
    config["block_K"] = hint.rstep[0]
    config["num_stages"] = hint.pipeline_stage
    config["thread_num"] = block_rows * block_cols * 32
    config["enable_rasteration"] = hint.rasterization_plan is not NoRasterization

```
