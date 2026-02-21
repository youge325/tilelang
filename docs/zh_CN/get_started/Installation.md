# 安装指南

## 通过 pip 安装

**通过 wheel 包或 PyPI 安装的前提条件：**

- **glibc**：2.28（Ubuntu 20.04 或更高版本）
- **Python 版本**：>= 3.8
- **CUDA 版本**：12.0 <= CUDA < 13

安装 tilelang 最简单的方式是直接通过 pip 从 PyPI 安装。在终端中运行以下命令即可安装最新版本：

```bash
pip install tilelang
```

或者，您也可以使用发布页面上提供的预构建包来安装 tilelang：

```bash
pip install tilelang-0.0.0.dev0+ubuntu.20.4.cu120-py3-none-any.whl
```

要从 GitHub 仓库安装最新版本的 tilelang，可以运行以下命令：

```bash
pip install git+https://github.com/tile-ai/tilelang.git
```

安装完成后，可以通过以下命令验证安装：

```bash
python -c "import tilelang; print(tilelang.__version__)"
```

## 从源码构建

**从源码构建的前提条件：**

- **操作系统**：Linux
- **Python 版本**：>= 3.8
- **CUDA 版本**：>= 10.0

如果您更倾向于使用 Docker，请直接跳转到 [使用 Docker 安装](#install-using-docker) 部分。本节重点介绍在原生 Linux 环境中从源码构建。

首先，在 Ubuntu/Debian 系统上安装操作系统级别的依赖：

```bash
apt-get update
apt-get install -y python3 python3-dev python3-setuptools gcc zlib1g-dev build-essential cmake libedit-dev
```

然后，克隆 tilelang 仓库并使用 pip 安装。`-v` 标志可以在构建过程中启用详细输出。

> **注意**：使用 `--recursive` 标志以包含必要的子模块。Tilelang 目前依赖于一个定制版本的 TVM，该版本作为子模块包含在内。如果您更倾向于 [使用已有的 TVM 安装](#using-existing-tvm)，可以跳过克隆 TVM 子模块（但仍然需要其他依赖项）。

```bash
git clone --recursive https://github.com/tile-ai/tilelang.git
cd tilelang
pip install . -v
```

如果您想以开发模式安装 tilelang，可以使用 `-e` 标志，这样对 Python 文件的任何更改都会立即生效，无需重新安装。

```bash
pip install -e . -v
```

> **注意**：对 C++ 文件的更改需要重新构建 tilelang C++ 库。请参阅下面的 [开发者快速重构建](#faster-rebuild-for-developers)。如果使用 `pip install`，会创建一个默认的 `build` 目录，因此您也可以直接在 `build` 目录中运行 `make` 来重新构建，如下面的 [通过 PYTHONPATH 从源码工作](#working-from-source-via-pythonpath) 所述。

(working-from-source-via-pythonpath)=

### 通过 `PYTHONPATH` 从源码工作（推荐开发者使用）

如果您更喜欢通过 `PYTHONPATH` 直接从源码树工作而不使用 pip，请确保先构建原生扩展（`libtilelang.so`）：

```bash
mkdir -p build
cd build
cmake .. -DUSE_CUDA=ON
make -j
```

我们还推荐使用 `ninja` 来加速编译：

```bash
cmake .. -DUSE_CUDA=ON -G Ninja
ninja
```

然后在导入 `tilelang` 之前，将仓库根目录添加到 `PYTHONPATH`，例如：

```bash
export PYTHONPATH=/path/to/tilelang:$PYTHONPATH
python -c "import tilelang; print(tilelang.__version__)"
```

一些有用的 CMake 选项：
- `-DUSE_CUDA=ON|OFF` 构建 NVIDIA CUDA 支持（当找到 CUDA 头文件时默认为 ON）。
- `-DUSE_ROCM=ON` 在 AMD GPU 上构建时选择 ROCm 支持。
- `-DNO_VERSION_LABEL=ON` 禁用 `tilelang.__version__` 中的后端/git 后缀。

(using-existing-tvm)=

### 使用自定义 TVM 路径构建

如果您已有 TVM 代码库，可以使用 `TVM_ROOT` 环境变量来指定已有 TVM 仓库的位置：

```bash
TVM_ROOT=<your-tvm-repo> pip install . -v
```

> **注意**：这仍然会重新构建 TVM 相关的库（存储在 `TL_LIBS` 中）。这种方式经常会导致一些路径问题。请检查 `env.py` 查看一些未正确设置的环境变量。

(install-using-docker)=

## 使用 Docker 安装

对于希望使用预配置了所有依赖项的容器化环境的用户，tilelang 为不同的 CUDA 版本提供了 Docker 镜像。这种方式特别适用于确保不同系统之间环境的一致性。

**前提条件：**
- 系统已安装 Docker
- 构建 tilelang 不需要 NVIDIA Docker 运行时或 GPU，您可以在没有 GPU 的主机上构建，然后在其他机器上使用构建好的镜像。

1. **克隆仓库**：

```bash
git clone --recursive https://github.com/tile-ai/tilelang
cd tilelang
```

2. **构建 Docker 镜像**：

进入 docker 目录并为所需的 CUDA 版本构建镜像：

```bash
cd docker
docker build -f Dockerfile.cu120 -t tilelang-cu120 .
```

可用的 Dockerfile：
- `Dockerfile.cu120` - 适用于 CUDA 12.0
- 其他 CUDA 版本可能在 docker 目录中提供

3. **运行 Docker 容器**：

启动具有 GPU 访问权限和卷挂载的容器：

```bash
docker run -itd \
  --shm-size 32g \
  --gpus all \
  -v /home/tilelang:/home/tilelang \
  --name tilelang_b200 \
  tilelang-cu120 \
  /bin/zsh
```

**命令参数说明：**
- `--shm-size 32g`：增加共享内存大小以提高性能
- `--gpus all`：启用对所有可用 GPU 的访问
- `-v /home/tilelang:/home/tilelang`：将主机目录挂载到容器（根据需要调整路径）
- `--name tilelang_b200`：为容器指定名称以便于管理
- `/bin/zsh`：使用 zsh 作为默认 shell

4. **进入容器并验证安装**：

```bash
docker exec -it tilelang_b200 /bin/zsh
# 在容器内：
python -c "import tilelang; print(tilelang.__version__)"
```

### ROCm 容器构建（gfx942/gfx950）

如果您需要一个可直接使用的、从源码构建 TileLang 的 ROCm 镜像，请使用
`docker/Dockerfile.rocm`。这是获得干净、可复现环境的推荐方式。

如果您已经在另一个 ROCm 容器内（例如 `sglang` 镜像），只需要就地重新构建 TileLang，请按照以下步骤操作。

如果您使用的是 `sglang` ROCm 容器，需要在其中构建 TileLang（例如在 MI300 `gfx942` 或 MI355 `gfx950` 上），构建需要额外的系统库、Cython 和有效的 `llvm-config`。以下步骤与 `sglang/docker/rocm.Dockerfile` 中使用的构建流程一致：

```bash
# 在容器内（以 root 身份）
apt-get update && apt-get install -y --no-install-recommends \
  build-essential git wget curl ca-certificates gnupg \
  libgtest-dev libgmock-dev \
  libprotobuf-dev protobuf-compiler libgflags-dev libsqlite3-dev \
  python3 python3-dev python3-setuptools python3-pip \
  gcc libtinfo-dev zlib1g-dev libedit-dev libxml2-dev \
  cmake ninja-build pkg-config libstdc++6 \
  && rm -rf /var/lib/apt/lists/*

# 优先使用容器虚拟环境（避免使用系统 pip）
export PATH="/opt/venv/bin:${PATH}"

# 构建 GoogleTest 静态库（Ubuntu 包仅提供源码）
cmake -S /usr/src/googletest -B /tmp/build-gtest -DBUILD_GTEST=ON -DBUILD_GMOCK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/build-gtest -j"$(nproc)"
cp -v /tmp/build-gtest/lib/*.a /usr/lib/x86_64-linux-gnu/
rm -rf /tmp/build-gtest

# 保持 setuptools < 80（兼容某些基础镜像）
pip install --upgrade "setuptools>=77.0.3,<80" wheel cmake ninja scikit-build-core

# 定位 ROCm llvm-config（如果缺失则安装 LLVM 18）
LLVM_CONFIG_PATH=""
for p in /opt/rocm/llvm/bin/llvm-config /opt/rocm/llvm-*/bin/llvm-config /opt/rocm-*/llvm*/bin/llvm-config; do
  if [ -x "$p" ]; then LLVM_CONFIG_PATH="$p"; break; fi
done
if [ -z "$LLVM_CONFIG_PATH" ]; then
  echo "未找到 ROCm llvm-config；正在安装 LLVM 18..."
  curl -fsSL https://apt.llvm.org/llvm.sh -o /tmp/llvm.sh
  chmod +x /tmp/llvm.sh
  /tmp/llvm.sh 18
  LLVM_CONFIG_PATH="$(command -v llvm-config-18)"
  if [ -z "$LLVM_CONFIG_PATH" ]; then
    echo "错误：安装后未找到 llvm-config-18"
    exit 1
  fi
fi
export LLVM_CONFIG="$LLVM_CONFIG_PATH"
export PATH="$(dirname "$LLVM_CONFIG"):/usr/local/bin:${PATH}"

# 可选：为需要 llvm-config-16 的工具创建垫片
mkdir -p /usr/local/bin
printf "#!/usr/bin/env bash\nexec \"%s\" \"\$@\"\n" "$LLVM_CONFIG_PATH" > /usr/local/bin/llvm-config-16
chmod +x /usr/local/bin/llvm-config-16

# TVM Python 部分需要 Cython（用于构建使用的系统 Python）
pip install --no-cache-dir "cython>=0.29.36,<3.0"

# 克隆 + 构建 TileLang（ROCm）
# 默认位置：/opt/tilelang（如果您喜欢其他路径，请调整）。
git clone --recursive https://github.com/tile-ai/tilelang.git /opt/tilelang
cd /opt/tilelang
git submodule update --init --recursive
export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG}"

# 通过跳过依赖解析来避免拉取 CUDA wheels / 重新安装 torch。
# 假设 torch 已经安装在容器中。
pip install -e . -v --no-build-isolation --no-deps

# 使用 --no-deps 时手动安装所需的运行时依赖。
# 注意：在 ROCm 上跳过 torch-c-dlpack-ext（其 wheel 需要 CUDA 库）。
pip install "apache-tvm-ffi>=0.1.6" "z3-solver>=4.13.0"
# 如果您已安装 torch-c-dlpack-ext 并遇到 `libtorch_cuda.so` 错误：
# pip uninstall -y torch-c-dlpack-ext

# 如果您遇到 Cython 编译错误，如 `PyLong_SHIFT`/`digit` 未声明，
# 请为可编辑构建禁用 stable ABI（abi3）：
# export CMAKE_ARGS="-DUSE_CUDA=OFF -DUSE_ROCM=ON -DROCM_PATH=/opt/rocm -DLLVM_CONFIG=${LLVM_CONFIG} -DSKBUILD_SABI_VERSION="
# pip install -e . -v --no-build-isolation --no-deps

# 验证
python -c "import tilelang; print(tilelang.__version__)"
```

如果您仍然想使用 `pip install -e . -v --no-build-isolation`（不带 `--no-deps`），pip 会尝试解析 TileLang 的依赖并可能下载 CUDA wheels（例如 `nvidia_cudnn`、`nvidia_nvshmem`）并重新安装 `torch`。为了在 ROCm 容器中避免这种情况，请保持使用 `--no-deps` 并确保所需的包已经安装。

## 安装 Nightly 版本

对于希望在正式发布前获取最新功能和改进的用户，我们提供 tilelang 的 nightly 构建版本。

```bash
pip install tilelang -f https://tile-ai.github.io/whl/nightly
# 或 pip install tilelang --find-links https://tile-ai.github.io/whl/nightly
```

> **注意：** Nightly 构建包含最新的代码更改，但可能不如正式版本稳定。它们非常适合测试新功能或需要特定尚未发布的 bug 修复的场景。

## 安装配置

### 构建时环境变量
`USE_CUDA`：是否启用 CUDA 支持，默认：Linux 上为 `ON`，设置为 `OFF` 可构建 CPU 版本。默认使用 `/usr/local/cuda` 来构建 tilelang。设置 `CUDAToolkit_ROOT` 以使用不同的 CUDA 工具包。

`USE_ROCM`：是否启用 ROCm 支持，默认：`OFF`。如果您的 ROCm SDK 不在 `/opt/rocm`，请设置 `USE_ROCM=<rocm_sdk>` 来针对自定义 SDK 路径启用 ROCm 构建。

`USE_METAL`：是否启用 Metal 支持，默认：Darwin 上为 `ON`。

`TVM_ROOT`：要使用的 TVM 源码根目录。

`NO_VERSION_LABEL` 和 `NO_TOOLCHAIN_VERSION`：
构建 tilelang 时，我们会尝试将 SDK 和版本信息嵌入到包版本号中，如下所示。
其中本地版本标签看起来像 `<sdk>.git<git_hash>`。设置 `NO_VERSION_LABEL=ON` 可禁用此行为。
```
$ python -mbuild -w
...
Successfully built tilelang-0.1.6.post1+cu116.git0d4a74be-cp38-abi3-linux_x86_64.whl
```

其中 `<sdk>={cuda,rocm,metal}`。具体来说，当 `<sdk>=cuda` 且通过环境变量提供了 `CUDA_VERSION` 时，
`<sdk>=cu<cuda_major><cuda_minor>`，与 PyTorch 中的对应部分类似。
设置 `NO_TOOLCHAIN_VERSION=ON` 可禁用此功能。

### 运行时环境变量

请参阅 `env.py` 文件获取支持的运行时环境变量的完整列表。

## 其他提示

### IDE 配置

在本地构建 tilelang 时，会自动在 `build` 目录中生成 `compile_commands.json` 文件。
安装了 clangd 和 [clangd 扩展](https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd) 的 VSCode 应该无需额外配置即可索引该文件。

### 编译缓存

编译缓存的默认路径为 `~/.tilelang/cache`。如果检测到 `ccache`，将自动使用。

### 修复 Wheel

如果您计划在其他环境中使用您的 wheel，
建议使用 auditwheel（Linux 上）或 delocate（Darwin 上）
来修复它们。

(faster-rebuild-for-developers)=

### 开发者快速重构建

`pip install` 会引入额外的打包/解包过程，即使没有源码更改也需要约 30 秒才能完成。

需要频繁重新编译的开发者可以使用：

```bash
pip install -r requirements-dev.txt

# 首次编译
pip install -e . -v --no-build-isolation

# 或使用 cmake/ninja 手动编译。记得正确设置 PYTHONPATH。
mkdir build
cd build
cmake .. -G Ninja
ninja

# 当您修改了 cpp 代码后重新构建
cd build; ninja
```

在可编辑/开发者模式下运行时，
您会看到类似以下的日志：

```console
$ python -c 'import tilelang'
2025-10-14 11:11:29  [TileLang:tilelang.env:WARNING]: Loading tilelang libs from dev root: /Users/yyc/repo/tilelang/build
```
