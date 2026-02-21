# 👋 欢迎来到 Tile Language

[GitHub](https://github.com/tile-ai/tilelang)

Tile Language（tile-lang）是一种简洁的领域特定语言，旨在简化高性能 GPU/CPU 内核（如 GEMM、Dequant GEMM、FlashAttention、LinearAttention）的开发。
通过采用 Python 风格的语法和基于 TVM 的底层编译器基础设施，tile-lang 让开发者能够专注于生产力，同时不牺牲达到最先进性能所需的底层优化。

:::{toctree}
:maxdepth: 2
:caption: 快速入门

get_started/Installation
get_started/overview
get_started/targets
:::

:::{toctree}
:maxdepth: 1
:caption: 教程

tutorials/debug_tools_for_tilelang
tutorials/auto_tuning
tutorials/logging
:::

:::{toctree}
:maxdepth: 1
:caption: 编程指南

programming_guides/overview
programming_guides/language_basics
programming_guides/instructions
programming_guides/control_flow
programming_guides/python_compatibility
programming_guides/autotuning
programming_guides/type_system
:::

:::{toctree}
:maxdepth: 1
:caption: 深度学习算子

deeplearning_operators/elementwise
deeplearning_operators/gemv
deeplearning_operators/matmul
deeplearning_operators/matmul_sparse
deeplearning_operators/deepseek_mla
:::

:::{toctree}
:maxdepth: 1
:caption: 编译器内部原理

compiler_internals/letstmt_inline
compiler_internals/inject_fence_proxy
compiler_internals/tensor_checks
:::

:::{toctree}
:maxdepth: 1
:caption: API 参考

autoapi/tilelang/index
:::

:::{toctree}
:maxdepth: 1
:caption: 隐私政策

privacy
:::
