# 张量检查（宿主端自动验证）

本页介绍 TileLang 自动插入到内核生成的宿主桩（host stub）中的宿主端检查。当你将 `torch.Tensor` 或任何兼容 DLPack 的对象传递给 TileLang 内核时，宿主桩会验证参数数量、指针类型、数据类型（dtype）、形状（shape）、步长（strides）、设备（device）等——这样你就不需要手动编写 Python 检查代码。这种方式保持了 ABI 的稳定性，并且与在 Python 中或通过 pybind 进行等效检查相比，显著降低了 Python 开销。

## 为什么需要宿主端检查
- ABI 稳定性：入口基于 TVM FFI + DLPack，统一接受张量和标量。
- 更低的开销：将检查从 Python 移至 C 端，减少了解释器/属性访问成本；调用开销低于基于 pybind 的方案。
- 精准的错误报告：断言在调用点附近触发，并给出精确的"哪个字段失败"的消息。

## 如何查看宿主源码
你可以查看自动生成的宿主源码（包含所有检查和最终的设备内核调用），用于调试：

```python
print(matmul_relu_kernel.get_host_source())
```

---

## 宿主检查内容

### 1) 参数数量和指针类型
- `num_args` 必须与形参数量匹配；否则内核返回 `-1` 并附带错误消息。
- 每个参数的 FFI 类型必须是指针类型（用于 DLTensor/handle）或有效的标量类型；否则你会看到类似 `Expect arg[i] to be pointer` 或标量类型错误的消息。

### 2) 张量检查（每个张量，在可空性判定之后）
- 可空性（Nullability）
  - 如果张量被函数体"静态可达/使用"，则句柄必须为非 NULL；否则报错：`xxx is expected to have non-NULL pointer`。
  - 如果输入张量未被函数使用（静态不可达），则允许 NULL；其他字段检查仅在 `handle != NULL` 时执行。
- 秩（ndim）
  - 运行时的 `ndim` 必须等于编译时的秩。
- 数据类型（dtype）
  - 匹配三元组 `(code, bits, lanes)`，并具有容差：
    - `float8_e4m3`：接受 `e4m3`、`e4m3fn`、`e4m3fnuz`。
    - `float8_e5m2`：接受 `e5m2`、`e5m2fnuz`。
    - `bool`：接受 `bits=8` 的 `int8/uint8`（lanes 相同）、`kDLBool(code=6, bits=1 或 8)`，以及任何 `bitwidth=1`（lanes 必须匹配）。
  - 对于打包位数据类型（例如 `Int(1)`、`Int(4)`、`UInt(4)`），跳过严格的 dtype 检查。
- 形状（Shape）
  - 每个运行时维度绑定到编译时形状（常量或符号），并检查一致性。
  - 符号维度之间的线性方程可在运行时即时求解（当某个检查点只有一个未知量时），从而支持跨张量约束。
- 步长（Strides）
  - 如果 `buffer_type = AutoBroadcast`：允许 `strides == NULL` 并从 `shape` 推导步长。如果存在显式的 `strides`，则绑定到编译时约束并检查相等性。
  - 否则：逐维检查；如果 `strides == NULL`，则从 `shape` 推导并进行比较（例如，连续布局：`strides[-1] == 1`，`strides[-2] == shape[-1]`）。
- `byte_offset`
  - 必须为 0（非零会引发错误），以保持寻址简单且对齐。
- 设备信息（Device info）
  - 断言 `device_type == 目标后端`（CUDA/ROCM/Metal/OneAPI/WebGPU/CPU 等）。错误消息包含 DLPack 代码对照表。
  - 当多个张量参与时，断言各张量的 `device_id` 一致。
- 数据指针（Data pointer）
  - 当可空性规则要求张量为非空时，数据指针必须为非 NULL。

### 3) 标量检查
- `T.int*` 系列：要求整数；错误：`Expect arg[i] to be int`。
- `T.bool`：要求布尔值；错误：`Expect arg[i] to be boolean`。

---

## 形状和符号方程：线性求解
当形状为符号（symbolic）形式时，宿主端会在运行时绑定并（在可能的情况下）求解线性关系（每个检查点只有一个未知量）。示例：

```python
@T.prim_func
def main(
    A: T.Tensor((m,), dtype),
    B: T.Tensor((m + n,), dtype),
    C: T.Tensor((n * k,), dtype),
):
    ...
```

这使得在运行时可以强制执行跨张量关系，如 `len(B) == m + n` 和 `len(C) == n * k`。

---

## 可空性规则和示例
哪些张量可以为 NULL？

- 规则：如果输入张量在静态分析下未被函数使用（即访问在静态上不可达），则视为可空（Nullable）；否则必须为非 NULL。
- 示例：

1) 必须为非 NULL（已使用）
```python
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    A[0] = 1
```
传入 `None` 会引发错误：`main.A_handle is expected to have non-NULL pointer`。

2) 仍然必须为非 NULL（常量为真的分支）
```python
some_cond: bool = True
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    if some_cond:
        A[0] = 1
```

3) 可空（常量为假的分支，静态不可达）
```python
some_cond: bool = False
@T.prim_func
def main(A: T.Tensor((M, K), dtype)):
    if some_cond:
        A[0] = 1
```

4) 必须为非 NULL（运行时条件）
```python
@T.prim_func
def main(A: T.Tensor((M, K), dtype), some_cond: T.bool):
    if some_cond:
        A[0] = 1
```
由于 `some_cond` 仅在运行时可知，静态分析无法证明 `A` 未被使用；因此 `A` 是不可空的。

---

## 设备类型代码（DLPack）
错误消息中支持和引用的设备代码：`1=CPU, 2=CUDA, 7=Vulkan, 8=Metal, 10=ROCM, 14=OneAPI, 15=WebGPU`。
内核会断言 `device_type` 与目标后端匹配，并要求各张量的 `device_id` 一致。

---

## 常见错误示例（你会看到的错误）
- 参数数量不匹配（num_args）
  - 触发条件：缺少或多余参数
  - 错误：`<kernel>: num_args should be N; expected: <num_args>, got: N`

- 期望指针类型参数
  - 触发条件：在期望张量的位置传入了标量
  - 错误：`<kernel>: Expect arg[i] to be pointer`

- 秩（ndim）不匹配
  - 触发条件：运行时秩与编译时秩不同
  - 错误：`<kernel>.<name>.ndim is expected to equal R, but got mismatched ndim`

- 数据类型（dtype）不匹配
  - 触发条件：dtype 与编译时 dtype 不同且不在容差范围内
  - 错误：`<kernel>.<name>.dtype is expected to be <dtype>, but got incompatible dtype`

- 形状约束违反
  - 触发条件：某维度不满足常量/符号绑定
  - 错误：`Argument <kernel>.<name>.shape[i] has an unsatisfied constraint: ... == <expected>`

- 步长检查失败（例如非连续布局）
  - 触发条件：转置/切片的张量违反了预期步长
  - 错误：`Argument <kernel>.<name>.strides[j] has an unsatisfied constraint: ... == <expected>`

- 设备类型不匹配
  - 触发条件：使用 CPU 张量调用 CUDA 内核等
  - 错误：`<kernel>.<name>.device_type mismatch [expected: <code> (<name>)] ...`

- 设备 ID 不匹配
  - 触发条件：混合使用来自不同 GPU 的张量
  - 错误：`Argument <kernel>.<name>.device_id has an unsatisfied constraint: ... == ...`

- NULL 数据指针
  - 触发条件：要求非空的张量具有 NULL 数据指针
  - 错误：`<kernel>.<name> is expected to have non-NULL data pointer, but got NULL`

- 标量类型不匹配
  - 触发条件：向 `T.int32` 传入浮点数，或向 `T.bool` 传入非布尔值
  - 错误：`<kernel>: Expect arg[i] to be int/boolean`

---

## 故障排除技巧
- 打印宿主源码：`print(fn.get_host_source())` 查看确切的断言以及预期值与实际值。
- 修复步长：对非连续张量调用 `.contiguous()`，或避免生成会打破假设的转置/切片布局。
- 对齐设备：确保所有参与的张量共享相同的 `device_type` 和 `device_id`。
- 对齐数据类型：使用 `.to(<dtype>)` 或以正确的 dtype 创建张量；注意 `float8` 和 `bool` 的容差规则。
- 动态形状：确保跨张量的线性关系在检查点处可以唯一确定（每次只有一个未知量）。

---

## 常见问题
- 可以禁用这些检查吗？
  - 不建议，通常也不支持。检查在宿主端完成，以保持 ABI 稳定性并在设备调用前尽早失败。
- 开销明显吗？
  - 检查是轻量级的（分支和字段读取）。与 Python 端检查相比更快；主要的开销仍然在 Python→C 的边界。总体上比等效的 Python 检查更经济。

---

## 参考示例（Matmul + ReLU）

```python
@T.prim_func
def matmul_relu_kernel(
    A: T.Tensor((M, K), dtype),
    B: T.Tensor((K, N), dtype),
    C: T.Tensor((M, N), dtype),
):
    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            T.copy(A[by * block_M, ko * block_K], A_shared)
            T.copy(B[ko * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])

# For debugging, print the host source
print(matmul_relu_kernel.get_host_source())
```

宿主端将为此示例插入上述所有检查。

---

## 快速错误参考
- 参数数量
  - 触发条件：缺少/多余参数；错误：`num_args should be N; expected: <num_args>, got: N`。
- 指针类型
  - 触发条件：向张量参数传入标量；错误：`Expect arg[i] to be pointer`。
- 秩（ndim）
  - 触发条件：运行时秩 != 编译时秩；错误：`ndim ... expected to equal R`。
- 数据类型（dtype）
  - 触发条件：不匹配且不在容差范围内；错误：`dtype ... expected to be <dtype>`。
- 形状
  - 触发条件：常量/符号绑定违反；错误：`shape[i] ... == <expected>`。
- 步长
  - 触发条件：布局不匹配；错误：`strides[j] ... == <expected>`。
- 设备类型
  - 触发条件：错误的后端设备；错误：`device_type mismatch [expected: ...]`。
- 设备 ID
  - 触发条件：张量位于不同 GPU 上；错误：`device_id ... == ...`。
- 数据指针
  - 触发条件：要求非 NULL 但为 NULL；错误：`non-NULL data pointer`。
- 标量类型
  - 触发条件：错误的标量类型；错误：`Expect arg[i] to be int/boolean`。

---

## 宿主错误故障排除（最小复现）

以下是常见宿主端错误的最小复现代码片段，假设目标为 CUDA 的内核如 `matmul_relu_kernel`，其参数如下：

```python
# Convention:
# A: float16 [M, K]
# B: float16 [K, N]
# C: float16 [M, N]
# Target: CUDA (device_type=2)
fn = matmul_relu_kernel  # your compiled function
M = N = K = 1024
```

如果你的内核不同，请相应调整 dtype/device。

### 0. 提示：打印宿主源码
```python
print(fn.get_host_source())
```

### 1. num_args 不匹配
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float16)
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
# Missing C
fn(A, B)
```
预期错误：`<kernel>: num_args should be 3; expected: <num_args>, got: 3`。

修复：按照签名传入所有参数。

### 2. 期望指针（张量）但传入了标量
```python
import torch

B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(1, B, C)
```
预期错误：`<kernel>: Expect arg[0] to be pointer`。

修复：传入兼容 DLPack 的张量（例如 torch.Tensor）。

### 3. ndim 不匹配
```python
import torch

A = torch.empty((M, K, 1), device='cuda', dtype=torch.float16)  # rank=3
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
预期错误：`<kernel>.A_handle.ndim is expected to equal 2, but got mismatched ndim`。

修复：确保运行时秩等于编译时秩。

### 4. dtype 不匹配
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float32)  # should be float16
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
预期错误：`<kernel>.A_handle.dtype is expected to be float16, but got incompatible dtype`。

修复：`A = A.to(torch.float16)` 或以正确的 dtype 创建张量。

### 5. 形状常量/符号不匹配
```python
import torch

A = torch.empty((M, K + 1), device='cuda', dtype=torch.float16)  # K mismatched
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A, B, C)
```
预期错误：`Argument <kernel>.A_handle.shape[i] has an unsatisfied constraint: ... == <expected>`。

修复：满足跨张量的线性约束和常量。

### 6. 步长检查失败（非连续）
```python
import torch

A = torch.empty((M, K), device='cuda', dtype=torch.float16)
A_nc = A.t()  # transpose -> non-contiguous
B = torch.empty((K, N), device='cuda', dtype=torch.float16)
C = torch.empty((M, N), device='cuda', dtype=torch.float16)
fn(A_nc, B, C)
```
预期错误：`Argument <kernel>.A_handle.strides[1] has an unsatisfied constraint: ... == 1`。

修复：传入 `A_nc.contiguous()` 或在内核中调整布局预期。

### 7. device_type 不匹配
```python
import torch

A = torch.empty((M, K), device='cpu', dtype=torch.float16)
B = torch.empty((K, N), device='cpu', dtype=torch.float16)
C = torch.empty((M, N), device='cpu', dtype=torch.float16)
fn(A, B, C)  # CUDA-targeted kernel
```
预期错误：`<kernel>.A_handle.device_type mismatch [expected: 2 (cuda)] ...`。

修复：将张量移动到 CUDA 设备。

### 8. device_id 不匹配（多 GPU）
```python
import torch

A = torch.empty((M, K), device='cuda:0', dtype=torch.float16)
B = torch.empty((K, N), device='cuda:1', dtype=torch.float16)
C = torch.empty((M, N), device='cuda:0', dtype=torch.float16)
fn(A, B, C)
```
预期错误：`Argument <kernel>.B_handle.device_id has an unsatisfied constraint: ... == ...`。

修复：将所有张量放在同一 GPU 上（例如 `cuda:0`）。

### 9. NULL 数据指针（高级）
这通常来自手动构造的 DLTensor/NDArray，或外部框架传入了未分配/已释放存储的对象。常规的 `torch.Tensor` 分配很少遇到此问题。

预期错误：`<kernel>.<name> is expected to have non-NULL data pointer, but got NULL`。

修复：确保底层存储有效；在 PyTorch 场景中，避免从无效的外部句柄构造张量。

### 10. 标量类型不匹配（int / bool）
```python
import tilelang.language as T

@T.prim_func
def scalar_check(x: T.int32, flag: T.bool()):
    T.evaluate(0)

scalar_check(1.0, True)  # x is float -> Expect arg[0] to be int
scalar_check(1, 2.5)     # flag is float -> Expect arg[1] to be boolean
```

修复：传入正确的标量类型，例如 `scalar_check(1, True)`。

---

## 结语
- 对照内核签名交叉检查"形状 / 步长 / 设备 / 数据类型"，以高效定位问题。
- 对于复杂的符号关系，打印宿主源码以确认绑定/求解顺序，然后相应调整运行时的形状/布局。
