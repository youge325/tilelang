Tilelang/TVM 中的日志系统
===================================================
<div style="text-align: left;">
<em>作者：</em> <a href="https://github.com/SiriusNEO">SiriusNEO</a>
</div>

## TVM 日志系统概述

Tilelang 目前使用 TVM 的日志系统。具体实现可以在以下位置找到：

- [include/tvm/runtime/logging.h](https://github.com/apache/tvm/blob/main/include/tvm/runtime/logging.h)：宏定义
- [src/runtime/logging.cc](https://github.com/apache/tvm/blob/main/src/runtime/logging.cc)：日志逻辑实现

设计风格受到 [Google glog](https://google.github.io/glog/stable/) 的启发。

## 日志类别

有三种主要的宏类型：

```c++
LOG(INFO) << "aaa";
DLOG(INFO) << "aaa";
VLOG(1) << "aaa";
```

- **LOG**：标准日志，保留在代码中用于在运行时显示不同级别的必要信息。大多数 Tilelang C++ 错误报告通过 `LOG(FATAL) << "error msg"` 实现。
- **DLOG**：调试日志，用于开发者的调试输出。DLOG 在构建时由 TVM_LOG_DEBUG 环境变量控制，并且在**Release 构建中通过死代码消除被移除**。
  - LOG(DEBUG) 和 DLOG 之间的关键区别在于这种构建时的消除。我们建议使用 DLOG 而非 LOG(DEBUG)，因为后者功能重叠且会被编译到 release 运行时中。
- **VLOG**：[详细日志（Verbose logging）](https://google.github.io/glog/stable/logging/#verbose-logging)，主要用于调试。其主要特点是可自定义的详细级别。例如，VLOG(n) 中的 n 可以是 1、2、3、4、5 或 6，支持复杂的跟踪需求。相比之下，LOG 和 DLOG 通常使用预定义的详细级别，如 INFO 和 DEBUG。
  - 在实际的 Tilelang 开发中，VLOG 使用较少。
  - TVM 的 VLOG 使用 DLOG 实现，因此继承了 DLOG 的特性。

其他有用的宏包括各种 **CHECK** 变体：

```c++
CHECK(cond) << "error msg";
DCHECK(cond) << "error msg";
ICHECK(cond) << "error msg";
```

实现将错误路由到 LogFatal：

```c++
#define CHECK(x)                                                \
  if (!(x))                                                     \
  ::tvm::runtime::detail::LogFatal(__FILE__, __LINE__).stream() \
      << "Check failed: (" #x << ") is false: "
```
- **DCHECK**：调试模式的 CHECK，仅在调试构建中编译
- **ICHECK**：内部检查（Internal Check），应存在于 Release 构建中。当 ICHECK 失败时，整个系统应报告错误。

## 日志详细级别

TVM 为 LOG 和 DLOG 定义了 5 个级别（相比 glog 增加了 DEBUG）：

```c++
#define TVM_LOG_LEVEL_DEBUG 0
#define TVM_LOG_LEVEL_INFO 1
#define TVM_LOG_LEVEL_WARNING 2
#define TVM_LOG_LEVEL_ERROR 3
#define TVM_LOG_LEVEL_FATAL 4
```

## 在 TileLang 开发中使用日志

### 指导原则

对于代码中的临时调试输出，没有限制（您甚至可以使用 std::cout）。只需记住在提交 PR 之前将其移除。

对于应保留在 Tilelang 代码库中的有意义的日志：

- 关键正确性检查：使用 ICHECK 并附带足够的错误信息，以便在出现问题时便于调试。
- 复杂 Pass 的调试：对于需要中间输出的 pass（如 LayoutInference），可能需要将来审查，使用 DLOG。
- 一般 INFO/WARNING 消息：使用标准 LOG。

### 在 Tilelang 中启用日志输出

要在运行时指定当前日志级别，需要设置环境变量 `TVM_LOG_LEVEL`。示例用法：

```c++
TVM_LOG_DEBUG=1 python3 code.py
```

这将为所有文件启用所有 DEBUG/INFO（级别 <= 1）日志。

#### TVM_LOG_DEBUG 规范的详细规则

解析逻辑在 `logging.cc` 中。参考：[HyperAI 知乎文章](https://zhuanlan.zhihu.com/p/1933106843468665163)。

使用 `TVM_LOG_DEBUG=<spec>` 启动 Python，其中 `<spec>` 是以逗号分隔的级别赋值列表，格式为 `<file_name>=<level>`。重要注意事项：

- 特殊文件名 DEFAULT 设置所有文件的 LOG 级别。
- `<level>` 可以设置为 -1 以禁用该文件的 LOG。
- `<file_name>` 是相对于 TVM 仓库 `src/` 目录的 C++ 源文件名（例如 .cc，而非 .h）。指定文件路径时 `src/` 前缀是可选的。

### 启用调试模式

要启用 DLOG/DCHECK，开发者需要首先在 Debug 模式下构建 Tilelang：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSE_CUDA=ON
```

Tilelang 的 CMake 逻辑会自动添加 `TVM_LOG_DEBUG` 宏，编译所有 DLOG 语句：

```cmake
target_compile_definitions(tilelang_objs PRIVATE "TVM_LOG_DEBUG")
```

然后还需要指定运行时环境变量。例如，要使用 `DLOG(INFO) << "xxx"` 进行调试，请使用 INFO 级别（1）运行代码：`TVM_LOG_DEBUG=1`。

:::{note}
   **重要**：有两个 TVM_LOG_DEBUG 变量。(1) 编译时宏：决定调试内容（如 DLOG）是否编译到 .so 文件中。在 C++ 源码中通过 #ifdef TVM_LOG_DEBUG 引用。使用 CMake 的 Debug 构建模式时自动启用。(2) 运行时环境变量：在运行时控制日志级别。TVM 为此变量提供了一个规范，允许控制每个文件的日志级别。

   这两者理想情况下应该有不同的名称，但 TVM 对两者使用了相同的名称，这可能会造成混淆。
:::
