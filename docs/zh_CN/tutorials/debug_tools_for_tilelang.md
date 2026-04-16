# 调试 Tile Language 程序

<div style="text-align: left;">
<em>作者：</em> <a href="https://github.com/LeiWang1999">Lei Wang</a>
</div>

## 概述

一个 Tile Language 程序（以下简称*程序*）通过以下几个阶段被转换为硬件可执行文件：

1. 用户编写 Tile Language 程序。
2. 程序经过多个 *Pass* 进行转换和优化（*lower* 阶段，参见 `tilelang/engine/lower.py`），最终产生中间表示（例如 CPU 的 LLVM 或 C，NVIDIA GPU 的 CUDA 等）。
3. 生成的代码由相应的编译器（例如 nvcc）编译为硬件可执行文件。

```{figure} ../../_static/img/overview.png
:width: 300
:alt: 编译过程概览
:align: center

```

在此过程中，用户可能会遇到大致三类问题：

- **生成问题**：Tile Language 程序无法生成有效的硬件可执行文件（即 lowering 过程中出错）。
- **正确性问题**：生成的可执行文件可以运行，但结果不正确。
- **性能问题**：可执行文件运行性能远低于预期的硬件理论极限。

本教程重点关注前两类问题——如何调试生成和正确性问题。性能调优通常需要使用供应商提供的性能分析工具（如 **Nsight Compute**、**rocProf** 等）进行更深入的硬件级分析，我们将在未来的材料中介绍。

下面，我们以矩阵乘法（GEMM）为例来演示如何编写和调试一个 Tile Language 程序。

## 矩阵乘法示例

在 **Tile Language** 中，您可以使用 **Tile Library** 来实现矩阵乘法。以下是一个完整示例：

```python
import tilelang
import tilelang.language as T

def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    # ...existing code...

# 1. 使用所需的维度定义内核（matmul）
func = matmul(1024, 1024, 1024, 128, 128, 32)

# 2. 将内核编译为 torch 函数
# ...existing code...
```

## 调试生成问题

TileLang 本质上执行*渐进式降级*。例如，一个 `T.copy` 可能首先被展开为 `T.Parallel`（参见 pass `LowerTileOP`），然后再次展开，最终得到可以转换为 CUDA C 代码的更低级语句。

```{figure} ../../_static/img/ir_transform_diagram.png
:width: 400
:alt: IR 转换图
:align: center

```

当代码生成失败时（例如出现编译错误），您**不**一定需要直接跳入 C++ pass 进行调试。相反，您可以先在 Python 中通过打印来检查中间表示（IR）。

例如，考虑一个简单的一维 `T.copy` 导致 lowering 过程失败的情况。以下代码片段展示了该问题的简化版本（基于社区 Issue #35）：

```python
@T.prim_func
def main(Q: T.Tensor(shape_q, dtype)):
    # ...existing code...
```

TileLang 的 lower 过程可能会产生如下错误：

```text
File "/root/TileLang/src/target/codegen_cuda.cc", line 1257
ValueError: Check failed: lanes <= 4 (8 vs. 4) : Ramp of more than 4 lanes is not allowed.
```

这表明在代码生成过程中的某处引入了不支持的向量化模式（8 lanes 的 ramp）。在深入底层 C++ 代码之前，先查看代码生成之前的 IR 会很有帮助。例如：

```python
device_mod = tir.transform.Filter(is_device_call)(mod)
# ...existing code...
```

## 调试正确性问题

有时，内核可以编译和运行但产生不正确的结果。在这种情况下，有两个主要策略可以帮助调试：

1. **使用后处理回调来检查或修改生成的 CUDA 代码。**
2. **使用内置的 `T.print` 调试原语在运行时检查值。**

### 生成源代码的后处理回调

在代码生成之后（在 codegen pass 中），TileLang 调用一个回调函数（如果已注册）来允许对生成的源代码进行后处理。在 `src/target/rt_mod_cuda.cc` 中：

```cpp
std::string code = cg.Finish();
if (const auto *f = Registry::Get("tilelang_callback_cuda_postproc")) {
    code = (*f)(code, target).operator std::string();
}
```

因此，通过注册一个名为 `tilelang_callback_cuda_postproc` 的 Python 函数，您可以拦截最终的 CUDA 代码字符串。例如：

```python
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.engine.callback import register_cuda_postproc_callback

@register_cuda_postproc_callback
def tilelang_callback_cuda_postproc(code, _):
    print(code) # 打印最终的 CUDA 代码
    code = "// modified by tilelang_callback_cuda_postproc\n" + code
    return code

kernel = tilelang.compile(matmul, target="cuda")
kernel_source = kernel.get_kernel_source()
print(kernel_source)
'''
// modified by tilelang_callback_cuda_postproc
#include "cuda_runtime.h"
...
'''
```

### 使用 `T.print` 进行运行时调试打印

TileLang 提供了一个内置的调试原语 `T.print`，用于在内核中进行打印。在 GPU 代码中使用时，请注意并发和线程同步问题。以下是一些在 TileLang 程序中打印缓冲区、变量和其他数据的示例。

1. **打印整个缓冲区**

```python
def debug_print_buffer(M=16, N=16):
    # ...existing code...
```

2. **条件打印**

```python
def debug_print_buffer_conditional(M=16, N=16):
    # ...existing code...
```

3. **打印线程索引或标量值**

```python
def debug_print_value_conditional(M=16, N=16):
    # ...existing code...
```

4. **打印 Fragment（寄存器文件）内容**

```python
def debug_print_register_files(M=16, N=16):
    # ...existing code...
```

5. **添加消息前缀**

```python
def debug_print_msg(M=16, N=16):
    # ...existing code...
```

输出消息将包含类似以下内容：

```text
msg='hello world' BlockIdx=(0, 0, 0), ThreadIdx=(0, 0, 0): 0
```

### TileLang 的可视化布局推断
**可视化布局推断**工具自动生成可视化图表，展示逻辑索引、线程 ID 和寄存器文件位置之间的映射关系。

当 TileLang 执行布局推断时，它会确定 fragment 缓冲区如何在各线程之间分布。可视化布局工具捕获这些信息并生成：
1. **文本输出**：布局映射的人类可读描述
2. **可视化图表**：显示线程到数据映射的彩色图

可视化布局推断工具通过 `TL_LAYOUT_VISUALIZATION_ENABLE` 和 `TL_LAYOUT_VISUALIZATION_FORMATS` pass 配置来控制。默认情况下，`TL_LAYOUT_VISUALIZATION_ENABLE` 是**禁用**的，以避免编译期间的性能开销。

启用后，`TL_LAYOUT_VISUALIZATION_FORMATS` 接受字符串值来控制输出格式：
- "txt"：仅文本输出（与默认相同）
- "all"：生成所有格式（TXT、PDF、PNG、SVG）
- "png"：仅生成 PNG 格式
- "pdf"：仅生成 PDF 格式
- "svg"：仅生成 SVG 格式
- "txt,svg"：在文本输出之外生成多种格式（逗号分隔）

"txt" 的输出消息将包含类似以下内容：
```
C_local inferenced layout:
  Shape: [32, 32] -> [8]
  Thread: _j // 16 * 64 + _i // 16 * 32 + _i % 8 * 4 + _j % 8 // 2
  Index:  [_j % 16 // 8 * 4 + _i % 16 // 8 * 2 + _j % 2]
```

## AutoDD：自动增量调试

当处理产生错误的复杂 TileLang 程序时，手动隔离 bug 可能非常繁琐。**AutoDD**（自动增量调试）是一个内置工具，可以自动将程序简化为重现特定错误所需的最少代码。

### 什么是增量调试？

增量调试（Delta Debugging）是一种自动化调试技术，它：
1. 接收一个触发 bug 的程序
2. 系统性地移除代码片段
3. 检查简化后的程序是否仍然触发相同的 bug
4. 产生重现 bug 的最小代码

AutoDD 使用概率分布驱动的增量调试（PDD）算法来实现高效最小化。

### 为什么使用 AutoDD？

- **大型代码库**：真实项目通常有数百行配置、辅助函数和日志记录
- **难以定位的错误**：错误消息可能指向 TVM/CUDA 内部而非您的 TileLang 代码
- **节省时间**：手动删除代码来隔离 bug 非常耗时

AutoDD 可以将 200+ 行的程序减少到仅 30 行，直接暴露根本原因。

### 基本用法

```bash
python -m tilelang.autodd <source_file> --err-msg "<error_message>" -o <output_file>
```

### 参数

| 参数 | 描述 |
|-----------|-------------|
| `source` | 输入 Python 源文件的路径 |
| `--err-msg` | 要匹配的错误消息（在 stdout 或 stderr 中搜索） |
| `-o, --output` | 最小化输出文件的路径 |
| `--backend` | 执行后端：`runner`（更快）或 `subproc`（更稳定），默认 `runner` |
| `--timeout` | 每个任务的超时时间（秒），默认 60 |
| `-j, --jobs` | 并行作业数，默认 1 |

### 示例

假设您有一个包含 GEMM 形状不匹配 bug 的复杂 TileLang 程序：

```python
# buggy_matmul.py（200+ 行）
@tilelang.jit
def buggy_matmul(M, N, K, block_M, block_N, block_K, ...):
    @T.prim_func
    def matmul_kernel(...):
        with T.Kernel(...) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug: 应该是 (block_K, block_N)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # ... 大量其他代码 ...
            T.gemm(A_shared, B_shared, C_local)  # 此处出错
    return matmul_kernel
```

运行 AutoDD 进行最小化：

```bash
python -m tilelang.autodd buggy_matmul.py --err-msg "Dimension mismatch" -o minimized.py -j 4
```

AutoDD 将产生一个最小化重现：

```python
# minimized.py（约 30 行）
import tilelang.language as T

def buggy_matmul(M, N, K, block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32, *args, **kwargs):
    @T.prim_func
    def matmul_kernel():
        with T.Kernel():
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_M, block_N), dtype)  # Bug 暴露！
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.gemm(A_shared, B_shared, C_local)
```

### AutoDD 工作原理

AutoDD 使用 AST（抽象语法树）分析配合多种重写规则：

1. **快速缩减器**：移除语句，简化 if/for 结构
2. **规范化器**：展开 with 语句，添加 `*args, **kwargs` 以保证兼容性
3. **简化器**：用常量替换表达式，简化函数调用
4. **慢速缩减器**：移除任意表达式，减少整数常量

### 提示

- **错误消息匹配**：使用错误输出中的唯一子字符串
- **超时**：对编译时间较长的程序增加 `--timeout`
- **并行作业**：使用 `-j 4` 或更高来加速最小化
- **后端**：如果 `runner` 不稳定，尝试 `--backend subproc`

### 完整示例

完整示例可在 `examples/autodd/` 中找到：
- `tilelang_buggy.py`：一个包含 bug 的复杂程序（约 200 行）
- `tilelang_minimized_expected.py`：AutoDD 后的预期输出（约 30 行）
- `README.md`：详细文档

## 总结

通过仔细检查最终代码生成之前的中间表示（IR）——并利用 `T.print` 进行运行时打印——可以快速诊断索引计算、复制逻辑或其他内核操作与预期行为的偏差。这种双管齐下的方法（检查 IR 转换和使用运行时打印）通常足以解决 TileLang 程序中的生成和正确性问题。

对于手动调试繁琐的复杂程序，**AutoDD** 提供了自动化增量调试，可以快速隔离重现 bug 的最小代码。

对于高级性能调优（例如分析内存带宽或占用率），可能需要更专业的性能分析工具，如 **Nsight Compute**、**rocProf** 或供应商特定的分析器。这些方面将在未来的文档中介绍。
