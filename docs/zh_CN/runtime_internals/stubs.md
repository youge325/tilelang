# CUDA/CUDART/NVRTC 存根

本文档描述了 TileLang 中 CUDA 相关库的存根（stub）机制。

## 目的

1. **CUDA 驱动（`cuda`）**：允许在没有 GPU 的系统上（如 CI/编译节点）导入 TileLang，通过在需要时才延迟加载 `libcuda.so` 来实现。
2. **运行时和编译器（`cudart`、`nvrtc`）**：解决 SONAME 版本不匹配问题，使单次构建可以跨不同 CUDA 版本工作。这通过复用 PyTorch 等框架已加载的 CUDA 运行时来实现。

## 实现

`src/target/stubs/` 中的存根实现了延迟加载机制：

- **延迟加载**：库仅在首次 API 调用时通过 `dlopen` 加载。
- **全局符号复用**：对于 `cudart` 和 `nvrtc`，存根首先检查全局命名空间（`RTLD_DEFAULT`）以使用任何已加载的符号（例如来自 PyTorch）。
- **版本支持**：处理不同 CUDA 版本之间的 ABI 差异（例如 CUDA 12 中 `cudaGraphInstantiate` 的变更）。

## 构建选项

`TILELANG_USE_CUDA_STUBS`（默认：`ON`）控制此行为。启用后，TileLang 链接这些存根而非系统 CUDA 工具包。
