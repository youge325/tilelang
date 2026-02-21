# 理解编译目标（Target）

TileLang 构建于 TVM 之上，TVM 依赖**编译目标（target）**来描述您要编译的目标设备。
编译目标决定了使用哪个代码生成器（CUDA、HIP、Metal、LLVM 等），并允许您传递
设备特定的选项，例如 GPU 架构标志。本页面总结了在编译 TileLang 程序时
如何选择和自定义编译目标。

## 常用目标字符串

TileLang 提供了一组常用目标；每个目标接受完整范围的 TVM 选项，因此您可以细调
生成的代码。最常用的选择如下：

| 基础名称 | 描述 |
| --------- | ----------- |
| `auto` | 按 CUDA → HIP → Metal 的顺序自动检测。在不同机器上运行相同脚本时非常有用。 |
| `cuda` | NVIDIA GPU。支持 `-arch=sm_80`、`-max_num_threads=1024` 等选项。 |
| `cutedsl` | NVIDIA CUTLASS/CuTe DSL 后端。需要 `nvidia-cutlass-dsl`。`cuda` 选项也可应用于此目标。 |
| `hip` | 通过 ROCm 使用的 AMD GPU。可附加 `-mcpu=gfx90a` 等选项。 |
| `metal` | Apple Silicon GPU（arm64 Mac）。 |
| `llvm` | CPU 执行；接受标准的 TVM LLVM 开关。 |
| `webgpu` | 浏览器 / WebGPU 运行时。 |
| `c` | 输出纯 C 源代码，供检查或自定义工具链使用。 |

要添加选项，在基础名称后用空格分隔追加即可。例如：

```python
target = "cuda -arch=sm_90"
kernel = tilelang.compile(func, target=target, execution_backend="cython")
# 或
@tilelang.jit(target=target)
def compiled_kernel(*args):
    return func(*args)
```

同样的约定适用于 HIP 或 LLVM（例如 `hip -mcpu=gfx940`、`llvm -mtriple=x86_64-linux-gnu`）。

### 高级用法：指定精确的硬件

当您已经知道确切的 GPU 型号时，可以在目标字符串中编码——通过 `-arch=sm_XX` 或
使用 TVM 预定义的目标标签（如 `nvidia/nvidia-h100`）。提供此详细信息对于
TileLang 的一般使用是可选的，但当启用 TVM 代价模型时（例如在自动调优期间）
会很有价值。代价模型使用额外的属性来做出更好的调度预测。如果您跳过此步骤
（或不使用代价模型），通用目标如 `cuda` 或 `auto` 完全没有问题。

TVM 目标注册表识别的所有 CUDA 计算能力如下所示。选择与您的 GPU 匹配的计算能力，
并将其追加到目标字符串中或使用相应的目标标签——例如 `nvidia/nvidia-a100`。

| 架构 | GPU（示例） |
| ------------ | ---------------- |
| `sm_20` | `nvidia/tesla-c2050`、`nvidia/tesla-c2070` |
| `sm_21` | `nvidia/nvs-5400m`、`nvidia/geforce-gt-520` |
| `sm_30` | `nvidia/quadro-k5000`、`nvidia/geforce-gtx-780m` |
| `sm_35` | `nvidia/tesla-k40`、`nvidia/quadro-k6000` |
| `sm_37` | `nvidia/tesla-k80` |
| `sm_50` | `nvidia/quadro-k2200`、`nvidia/geforce-gtx-950m` |
| `sm_52` | `nvidia/tesla-m40`、`nvidia/geforce-gtx-980` |
| `sm_53` | `nvidia/jetson-tx1`、`nvidia/jetson-nano` |
| `sm_60` | `nvidia/tesla-p100`、`nvidia/quadro-gp100` |
| `sm_61` | `nvidia/tesla-p4`、`nvidia/quadro-p6000`、`nvidia/geforce-gtx-1080` |
| `sm_62` | `nvidia/jetson-tx2` |
| `sm_70` | `nvidia/nvidia-v100`、`nvidia/quadro-gv100` |
| `sm_72` | `nvidia/jetson-agx-xavier` |
| `sm_75` | `nvidia/nvidia-t4`、`nvidia/quadro-rtx-8000`、`nvidia/geforce-rtx-2080` |
| `sm_80` | `nvidia/nvidia-a100`、`nvidia/nvidia-a30` |
| `sm_86` | `nvidia/nvidia-a40`、`nvidia/nvidia-a10`、`nvidia/geforce-rtx-3090` |
| `sm_87` | `nvidia/jetson-agx-orin-32gb`、`nvidia/jetson-agx-orin-64gb` |
| `sm_89` | `nvidia/geforce-rtx-4090` |
| `sm_90a` | `nvidia/nvidia-h100`（DPX 配置文件） |
| `sm_100a` | `nvidia/nvidia-b100` |

请参阅 NVIDIA 的 [CUDA GPUs](https://developer.nvidia.com/cuda-gpus) 页面或 TVM 源代码
（`3rdparty/tvm/src/target/tag.cc`）获取设备与计算能力之间的最新映射。

## 以编程方式创建编译目标

如果您更喜欢使用 TVM 的 `Target` 对象，TileLang 暴露了辅助函数
`tilelang.utils.target.determine_target`（默认返回规范目标字符串，当
`return_object=True` 时返回 `Target` 对象）：

```python
from tilelang.utils.target import determine_target

tvm_target = determine_target("cuda -arch=sm_80", return_object=True)
kernel = tilelang.compile(func, target=tvm_target)
```

您也可以直接通过 TVM 构建目标：

```python
from tvm.target import Target

target = Target("cuda", host="llvm")
target = target.with_host(Target("llvm -mcpu=skylake"))
```

TileLang 接受 `str` 或 `Target` 输入；内部会使用规范字符串表示进行归一化和缓存。
**在用户代码中，我们强烈建议传递目标字符串而非 `tvm.target.Target` 实例——字符串使缓存键紧凑且在运行间确定性一致，而构造新的 `Target` 对象可能导致略高的哈希开销或不一致的身份语义。**

## 在代码中查看支持的目标

想要快速查看内置的基础名称及其描述？使用：

```python
from tilelang.utils.target import describe_supported_targets

for name, doc in describe_supported_targets().items():
    print(f"{name:>6}: {doc}")
```

此辅助函数反映了上面的表格，可以安全地在运行时调用（例如在验证 CLI 参数时）。

## 故障排除提示

- 如果您看到 `Target cuda -arch=sm_80 is not supported`，请仔细检查拼写并确认该选项对 TVM 是有效的。任何无效的开关都会显示为目标构造错误。
- 运行时错误如 "no kernel image is available" 通常意味着 `-arch` 标志与您正在运行的 GPU 不匹配。尝试去掉该标志或切换到正确的计算能力。
- 当面向多个环境时，使用 `auto` 以获得便利性，仅在需要架构特定调优时覆盖为显式字符串。
