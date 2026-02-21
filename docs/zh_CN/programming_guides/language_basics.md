# 语言基础

本页介绍 TileLang（tile-lang）DSL 的核心内容，您将使用这些来编写
高性能内核。重点在于如何定义内核、表达迭代、在内存作用域之间移动数据，以及使用 JIT 运行内核。

示例使用以下常规别名：

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## 1. 使用 `@T.prim_func` 定义内核

TileLang 内核是由 `@T.prim_func` 装饰器产生的 TIR（TVM IR）函数。
参数通过 `T.Tensor` 或 `T.Buffer` 标注形状和数据类型。

关于数据类型的说明
- 数据类型可以作为字符串传递（例如 'float32'）、TileLang 数据类型（例如 `T.float32`）
  或框架数据类型（例如 `torch.float32`）。TileLang 会将所有这些进行归一化。
  详见类型系统。

```python
@T.prim_func
def add_kernel(
    A: T.Tensor((N,), dtype),    # dtype 可以是 'float32' | T.float32 | torch.float32
    B: T.Tensor((N,), dtype),
    C: T.Tensor((N,), dtype),
):
    ...  # 内核体
```

- 形状可以是具体整数或符号化的。对于符号化的，您可以通过外层的 `@jit` 包装器传递
  Python 整数（如下所示），或者当需要命名符号维度时使用 `T.dyn` 标注。

```python
# 命名符号维度（可选）
K = T.dyn['K']
@T.prim_func
def uses_dyn(A: T.Tensor((K,), 'float32')):
    ...
```

### 动态符号维度：两种方式

TileLang 支持两种互补的方式来引入符号化（动态）维度：

- 通过 `T.dyn[...]` 的类型级标注（推荐用于函数签名）
  - 在 `T.Tensor((T.dyn['K'], ...), dtype)` 中使用，或绑定一次后重复使用（如上）。
  - 在内核体内部，建议从缓冲区的形状读取，例如 `M = A.shape[0]`。

- 通过 `T.dynamic(name, dtype)` 的项级变量
  - 创建一个可以直接在表达式/循环中使用的 TIR `tir.Var`。
  - 当您需要在体内引用维度符号时很方便。

```python
# 1) 仅标注符号；通过 shape 读取绑定的大小
K = T.dyn['K']  # dtype 默认为 int32
@T.prim_func
def foo(A: T.Tensor((K,), 'float32')):
    N = A.shape[0]
    for i in T.serial(N):
        ...

# 2) 显式 Var 符号，可在体内使用
K = T.dynamic('K', 'int32')   # 或 T.dynamic('K') 默认为 int32
@T.prim_func
def bar(A: T.Tensor((K,), 'float32')):
    for i in T.serial(K):
        ...
```

注意事项
- `T.symbolic(name, dtype)` 是 `T.dynamic` 的已弃用别名；建议使用 `T.dynamic`。
- 在 `@jit` 下，具体大小来自首次调用时的实际张量参数。
- 标注中的符号不需要作为单独的内核参数；TileLang 从参数形状绑定它们。

## 2. 使用 `T.Kernel` 启动工作

`with T.Kernel(...)` 声明一个启动上下文并创建块/线程绑定。
对于 GPU 后端，指定网格和每块线程数。

```python
with T.Kernel(grid_x, grid_y, threads=128) as (bx, by):
    ...  # bx/by 是 blockIdx.x/y
```

您很少需要原始线程索引；大多数内核在 `T.Kernel` 内部使用结构化循环
（`T.serial`、`T.unroll`、`T.Parallel`、`T.Pipelined`）。

## 3. 循环和控制流

核心循环构造映射到常见的硬件模式：

- `T.serial(start, stop[, step])`：普通 for 循环
- `T.unroll(start, stop[, step])`：展开循环
- `T.Parallel(ext0, ext1, ...)`：嵌套并行循环（适合逐元素操作）
- `T.Pipelined(iters, num_stages=N)`：生产者/消费者的软件流水线

```python
for i in T.serial(N):
    ...

for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

for k in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    # 跨阶段重叠复制/计算
    ...
```

条件判断使用标准的 Python `if`/`else`。当 tile 大小不能整除问题大小时，
使用谓词来守护边界。

## 4. 内存作用域和分配

TileLang 暴露了关键的软件管理作用域：

- 全局（Global）：设备内存（`T.Tensor` 参数的默认作用域）
- 共享（Shared）：片上，块可见（`T.alloc_shared(shape, dtype)`）
- Fragment 和标量：每线程的 fragment 和标量变量，但以共享视图表示
  （`T.alloc_fragment`、`T.alloc_var`）

```python
A_shared = T.alloc_shared((BM, BK), 'float16')
B_shared = T.alloc_shared((BK, BN), 'float16')
C_local  = T.alloc_fragment((BM, BN), 'float32')
T.clear(C_local)  # 清零累加器
```

## 5. 移动数据：`T.copy`

使用 `T.copy(src, dst)` 在作用域之间移动 tile。它接受缓冲区、
缓冲区区域或缓冲区加载；范围会被推断或可以广播。

```python
# 全局 -> 共享（tile 复制），范围从 dst 推断
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[ko * BK, bx * BN], B_shared)

# Fragment -> 全局（写回）
T.copy(C_local, C[by * BM, bx * BN])
```

`T.copy` 在编译期间执行合并访问和作用域特定的降级。

## 6. 最小端到端示例（向量加法）

```python
import tilelang
import tilelang.language as T
from tilelang import jit

@jit  # 在首次调用时从张量推断目标
def add(N: int, block: int = 256, dtype: str = 'float32'):

    @T.prim_func
    def add_kernel(
        A: T.Tensor((N,), dtype),
        B: T.Tensor((N,), dtype),
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
            for i in T.Parallel(block):
                gi = bx * block + i
                # 可选 — LegalizeSafeMemoryAccess 在访问可能越界时插入守护
                C[gi] = A[gi] + B[gi]

    return add_kernel

# 宿主端（显示 PyTorch；也支持 NumPy/DLPack）
import torch
N = 1 << 20
A = torch.randn(N, device='cuda', dtype=torch.float32)
B = torch.randn(N, device='cuda', dtype=torch.float32)
C = torch.empty(N, device='cuda', dtype=torch.float32)

kernel = add(N)
kernel(A, B, C)  # 在 GPU 上运行
torch.testing.assert_close(C, A + B)
```

注意事项
- `@jit` 包装器在首次编译后返回一个可调用的内核。
- 您可以通过外层 Python 函数传递编译时可调参数（tile 大小、数据类型），
  并将它们固化到生成的 TIR 中。

## 7. 分块 GEMM 骨架

以下是使用共享内存暂存和 fragment 累加器的分块 GEMM 最小模式。
它反映了仓库中的快速入门风格。

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
            T.copy(A[by * BM, ko * BK], A_s)
            T.copy(B[ko * BK, bx * BN], B_s)
            T.gemm(A_s, B_s, C_f)  # 降级为张量核心/ISA 特定内核

        T.copy(C_f, C[by * BM, bx * BN])
```

## 8. 调试和打印

在内核中使用 `T.print` 进行快速自省。TileLang 对共享/fragment 作用域
从单个线程发出打印以避免输出洪流。

```python
T.print(C_f, msg='accumulator:')
T.print(A_s, msg='A tile:')
T.print(C[0], msg='C[0] = ')
```

## 9. 下一步

- 控制流详情：参见编程指南 → 控制流
- 内存主题：基础内容已在行内涵盖
- 自动调优 tile 大小和映射：编程指南 → 自动调优
- 算子示例（GEMM、GEMV、attention）：参见深度学习算子
