# 控制流

本指南涵盖 TileLang 中的控制流原语以及它们如何降级为
高效的 GPU 代码。您将使用这些来组织循环、处理边界
和表达流水线计算。

## 概述
- 条件判断：`if` / `elif` / `else`，三元表达式（`x if c else y`）
- 循环：`T.serial`、`T.unroll`、`T.Parallel`、`T.Pipelined`
- While 循环：使用 TIR 条件的 `while`
- 流控制：Python `break` / `continue`
- 安全性：通过 LegalizeSafeMemoryAccess pass 自动进行越界保护

示例假设 `import tilelang.language as T`。

## 条件判断

在 `@T.prim_func` 内核中支持标准的 Python `if`/`elif`/`else`。
条件应为 TIR 表达式（例如 `i < N`）。Python 的普通布尔值被
视为编译时常量并会被折叠。

```python
for i in T.serial(N):
    if i < N:            # TIR 条件
        C[i] = A[i] + B[i]
    else:
        pass

# 三元表达式
x = (A[i] if i < N else 0)
```

支持短路布尔运算。对于多维边界，使用
`T.any_of` / `T.all_of` 以提高清晰度：

```python
if T.all_of(i < M, j < N):
    C[i, j] = A[i, j] + B[i, j]
```

边界处理说明
- LegalizeSafeMemoryAccess pass 在访问可能越界时自动插入守护，
  并在证明安全时省略守护。您通常可以在简单的边界处理中省略
  显式的 `if` 检查，但在需要自定义逻辑或清晰性时请保留它们。

## 循环

### Serial

`T.serial` 创建一个普通的 for 循环。常见形式：

```python
for i in T.serial(N):
    ...                     # 0..N-1

for i in T.serial(0, N, 2):
    ...                     # 0, 2, 4, ...
```

### Unroll

`T.unroll` 请求对小循环次数进行循环展开。

```python
for k in T.unroll(K_TILE):
    acc += a[k] * b[k]
```

高级：TileLang 将展开提示转发到 TIR；factor/显式旋钮可用于专家调优。

### Parallel（逐元素）

`T.Parallel(ext0, ext1, ...)` 构建嵌套循环，非常适合逐元素
操作。循环体在一个 `for` 头部接收所有索引：

```python
for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]
```

可选提示：
- `coalesced_width=` 控制内存合并宽度（用于向量化检查）。
- `loop_layout=` 接受 `T.Fragment` 来标注整个嵌套
  并行循环的布局。标注仅附加到最外层循环，
  且 `InputDim == 嵌套并行维度数`。

### Pipelined（软件流水线）

`T.Pipelined(iters, num_stages=...)` 重叠生产者/消费者阶段（例如，
全局→共享复制与计算）。这是 GEMM/attention 流水线的骨干。

```python
for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_s)  # 阶段：复制 A tile
    T.copy(B[ko * BK, bx * BN], B_s)  # 阶段：复制 B tile
    T.gemm(A_s, B_s, C_f)             # 阶段：计算
```

### Persistent（高级）

`T.Persistent(domain, wave_size, index, group_size=...)` 暴露持久化
线程块风格的循环。这是一个高级构造，TileLang 在后续 pass 中降级，
通常由专门的模板使用。

## While 循环

当条件为 TIR 表达式时支持 `while`。避免无限循环；
TileLang 在检测到常量为真的条件时会报错。

```python
i = 0
while i < N:
    ...
    if done:
        break
    i += 1
```

## Break 和 Continue

在 `T.serial`/`T.unroll`/`T.Parallel`/`while` 循环中使用 Python `break`/`continue`
来退出或跳过。为了可读性，请保持 `break`/`continue` 后的循环体整洁；
编译器会忽略死路径。

## 综合示例：残余 Tile 处理

以下是 2D 内核的典型边界模式。使用 LegalizeSafeMemoryAccess 时，
当您不需要自定义边界路径时可以省略显式的守护。

```python
for i, j in T.Parallel(M, N):
    gi = by * BM + i
    gj = bx * BN + j
    if T.all_of(gi < M, gj < N):     # 在很多情况下可选
        C[gi, gj] = A[gi, gj] + B[gi, gj]
```

## 调试条件

使用 `T.print` 在谓词下检查值。对于缓冲区，TileLang 从单个线程打印
以避免重复输出。

```python
if i == 0:
    T.print(C, msg='C tile:')
```
