# Python 兼容性

TileLang 是一种嵌入 Python 的 DSL，但并非所有 Python 语法都在
TileLang DSL 内部得到支持。本指南阐明了哪些可以使用，哪些不行，以及
如何将常见的 Python 模式转换为 TileLang 等效写法。特别地，我们关注
内核部分（`with T.Kernel` 内部的脚本）的语义。关于使用即时编译（eager-style JIT）
时的宿主端（host-side）语义，请期待我们即将发布的文档。

以下代码使用常规别名：

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

## 控制流与循环

| Python 特性          | 是否支持 | 说明 / 替代方案                      |
|-------------------------|:---------:|------------------------------------------|
| `for i in range(n)`     | ✅        | 映射为 `T.serial(n)`                    |
| `for i in range(a,b,s)` | ✅        | 映射为 `T.serial(a, b, s)`              |
| `for x in list`         | ❌        | 使用基于索引的循环                     |
| `while condition`       | ✅        |                                          |
| `if` / `elif` / `else`  | ✅        |                                          |
| `x if cond else y`      | ✅        | 三元表达式                       |
| `break` / `continue`    | ✅        |                                          |
| `enumerate()` / `zip()`    | ❌     |                                          |

## 数据访问

| Python 特性          | 是否支持 | 说明 / 替代方案                      |
|-------------------------|:---------:|------------------------------------------|
| `a[i]` 索引         | ✅        | 支持多维索引：`a[i, j, k]` |
| `a[i:j]` 切片        | ✅        | 创建 `BufferRegion`                   |
| `a[-1]` 负数索引  | ✅        |                                          |

## 赋值与算术运算

| Python 特性          | 是否支持 | 说明 / 替代方案                      |
|-------------------------|:---------:|------------------------------------------|
| `x = expr`              | ✅        |                                          |
| `+`、`-`、`*`、`/`、`%` | ✅        | 映射为设备端算术运算 |
| `+=`、`-=`、`*=` 等  | ✅        | 增量赋值                     |
| `a = b = c`             | ❌        | 使用单独的赋值语句                 |

## 函数与类

作为内核脚本语言，TileLang 不支持函数或类。您可以使用 `@T.macro` 来定义可复用的代码块，它们会在编译时被内联展开，类似于 CUDA 中的 `__device__` 函数。

## 语句与内置函数

| Python 特性          | 是否支持 | 说明 / 替代方案                      |
|-------------------------|:---------:|------------------------------------------|
| `with`                  | ⚠️        | 仅限 `T.Kernel`、`T.ws`                  |
| `assert`                | ⚠️        | 使用 `T.device_assert` 或 `T.assert`      |
| `print()`               | ⚠️        | 使用 `T.print()`；`print` 对 Python 表达式有效 |
| `len()`                 | ❌        | 使用 `buffer.shape[dim]`                  |
| `type()`、`isinstance()`| ❌        |                                          |
