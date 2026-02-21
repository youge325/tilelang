# 类型系统

本页列出了 TileLang 支持的数据类型及其在内核中的指定方式。
完整详情和权威列表请参阅 API 参考文档
（`autoapi/tilelang/index`）和 `tilelang.language.v2.dtypes`。

如何指定 dtype
- 使用以下任意形式；TileLang 内部会自动标准化：
  - 字符串：`'float32'`、`'int8'`、`'bfloat16'` 等
  - TileLang dtype 对象：`T.float32`、`T.int8`、`T.bfloat16` 等
  - 框架 dtype：`torch.float32`、`torch.int8`、`torch.bfloat16` 等

常见标量类型
- 布尔：`bool`
- 有符号整数：`int8`、`int16`、`int32`、`int64`
- 无符号整数：`uint8`、`uint16`、`uint32`、`uint64`
- 浮点：`float16`（半精度）、`bfloat16`、`float32`、`float64`

Float8 及低精度系列
- Float8：`float8_e3m4`、`float8_e4m3`、`float8_e4m3b11fnuz`、`float8_e4m3fn`、
  `float8_e4m3fnuz`、`float8_e5m2`、`float8_e5m2fnuz`、`float8_e8m0fnu`
- Float6：`float6_e2m3fn`、`float6_e3m2fn`
- Float4：`float4_e2m1fn`

向量化元素类型（SIMD 打包）
- 对于许多基本类型，可通过 lane 数量获得向量打包变体：
  `x2`、`x4`、`x8`、`x16`、`x32`、`x64`。
- 示例：
  - 整数：`int8x2`、`int8x4` 等，`int32x2`、`int32x4` 等
  - 无符号：`uint8x2`、`uint8x4` 等
  - 浮点：`float16x2`、`float16x4`、`float32x2`、`float32x4` 等
  - Float8/6/4 系列也提供 `x2/x4/x8/x16/x32/x64` 变体（如适用），
    例如 `float8_e4m3x2`、`float8_e4m3x4`、`float6_e2m3fnx8`、`float4_e2m1fnx16`。

注意事项
- 某些低精度格式（float8/6/4）的可用性取决于目标架构
  和后端支持。
- 请为混合精度计算显式选择累加数据类型（例如，
  使用 `float16` 输入和 `float32` 累加器的 GEMM）。
- 完整的最新列表可在 `tilelang.language.v2.dtypes` 中查看，
  并在 API 参考文档中呈现。
