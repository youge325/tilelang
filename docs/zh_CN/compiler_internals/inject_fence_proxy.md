# InjectFenceProxy Pass

`tl.InjectFenceProxy` 是一个 TIR 级别的变换，通过在控制流从通用内存操作切换到异步代理操作时插入 `fence.proxy.async` 指令，来保持 NVIDIA Hopper（SM90+）上的 GPU 代理状态一致性。

## 为什么需要围栏（Fence）

Hopper 将内存指令分为通用路径和异步代理路径。当一条异步指令（例如 `cp.async` 或 `tma.load`）在通用流量（如 `ldmatrix` 或普通缓冲区存储）之后发出时，硬件要求一个 `fence.proxy.async` 来保证顺序。缺少围栏可能导致竞态条件或未定义行为。

## Pass 的功能

- 遍历 `PrimFunc` 中的每条语句，跟踪其行为属于**通用（generic）**、**异步（async）**还是**中性（neutral）**代理（中性语句重置状态，例如显式围栏）。
- 自动将 `tma_store` 内置函数降级为所需的 `arrive`/`wait` 握手，使 TMA 存储正确参与同步。
- 在通用语句后面紧跟异步语句且中间没有中性屏障时，注入显式的 `fence.proxy.async`。

该 pass 是保守的：未知的外部调用被视为异步，以确保插入围栏而非意外遗漏。

### 时间线视图

```
generic initialize_wgmma_descriptor → generic shared-store → async wgmma
             │                           │                   │
             └─ generic proxy            ┴─ generic proxy    ┴─ async proxy
                         │        fence inserted here   ↑
                         └──────────────────────────────┘
```

代理跟踪器从左到右扫描序列。当检测到从通用到异步的转换（在上面的存储和 `cp.async` 之间）时，它会在异步路径运行之前合成一个 `fence.proxy.async` 来重置硬件代理状态。

## 内置函数覆盖范围

跟踪器理解 TileLang 的 TMA 加载/存储、共享内存 MMA（`wgmma`）和 TVM/PTX 异步复制（`cp.async` 变体）的内置函数。通用操作目前包括 `ldmatrix`、`stmatrix` 和描述符初始化。其他 IR 节点（循环、块、属性）从其主体推导代理类型，以使分析在结构化控制流中存活。

## 用法

该 pass 是 TileLang 默认降级流水线的一部分。要手动应用：

```python
from tilelang import tl
from tvm import IRModule

mod = IRModule({"main": prim_func})
with tvm.transform.PassContext():
    mod = tl.transform.InjectFenceProxy()(mod)
```

## 端到端示例

pass 之前：

```python
@T.prim_func
def kernel():
    with T.Kernel(1):
        desc = T.decl_buffer((1,), "uint64", scope="local.descriptor")
        smem = T.decl_buffer((128,), "float16", scope="shared")
        T.initialize_wgmma_descriptor(desc, T.uint64(0), 2, 1, 32)
        smem[0] = T.float16(0)
        T.ptx_wgmma_ss(
            "float16",
            "m64n64k16",
            T.bool(True),
            T.bool(True),
            "fp16",
            "fp16",
            "fp16",
            desc.data,
            T.int32(0),
            desc.data,
            T.int32(0),
            smem.data,
            T.int32(0),
            T.bool(True),
            1,
            1,
        )
```

经过 `tl.transform.InjectFenceProxy` 之后：

```python
@T.prim_func
def kernel():
    with T.Kernel(1):
        desc = T.decl_buffer((1,), "uint64", scope="local.descriptor")
        smem = T.decl_buffer((128,), "float16", scope="shared")
        T.initialize_wgmma_descriptor(desc, T.uint64(0), 2, 1, 32)
        smem[0] = T.float16(0)
        T.fence_proxy_async()
        T.ptx_wgmma_ss(
            "float16",
            "m64n64k16",
            T.bool(True),
            T.bool(True),
            "fp16",
            "fp16",
            "fp16",
            desc.data,
            T.int32(0),
            desc.data,
            T.int32(0),
            smem.data,
            T.int32(0),
            T.bool(True),
            1,
            1,
        )
```

唯一的变化是在通用描述符设置/共享内存写入和异步 `wgmma` 之间添加了 `fence_proxy_async`。在更大的内核中，该 pass 对嵌套块、循环和条件分支执行相同的操作。

## 扩展该 Pass

如果您引入了一个行为类似异步代理的新内置函数，请将其添加到 `src/transform/inject_fence_proxy.cc` 中的 `IsAsyncIntrinsic`。同样，为额外的通用操作扩展 `IsKnownGeneric`。添加新的中性屏障时，确保它们将代理类型设置为 `kNeutral`，以正确重置状态。
