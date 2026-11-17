# 入门

[English](../en/getting-started.md)

本页带你从一次克隆走到一个被仿真的 kernel。不需要 GPU：仓库自带示例 capture，capture 之后的所有阶段都在 CPU 上跑。

## 1. 你需要什么

- Linux，C++20 编译器，CMake 3.20 或更新，Python 3.10 或更新，以及 OpenSSL 开发头文件（libcrypto）。
- `PATH` 上的 `zstd` 命令：trace 以压缩形式存放。
- GPU 只在你想自己采集 kernel 时才需要（第 5 节）。

项目构建与测试所用的版本列在 [environment.md](environment.md)。

## 2. 构建与安装

在仓库根目录：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure    # 可选：跑 C++ 测试
pip install -e .
```

构建在 `build/` 下产出四个二进制：`cyclescope-engine`（仿真器）与 `cyclescope-capture-facts`、`cyclescope-convert`、`cyclescope-trace-split`（转换工具）。`pip install -e .` 加上 `cyclescope` 命令，由它驱动这些工具。

像上面这样从克隆做可编辑安装。本包从源码检出读取 `configs/`、`suites/` 与 `scripts/`，所以没有源码树的裸 `pip install` 不可行，命令会如实提示。

不安装只想试试命令：

```bash
PYTHONPATH=python python3 -m cyclescope.cli --help
```

## 3. 仿真你的第一个 kernel

`examples/captures/` 存了十个采集好的 kernel，每架构五个。跑一个：

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example
cyclescope report out/example
```

`cyclescope run` 把这个 capture 转换、构建成作业文件并仿真；`cyclescope report` 每个用例打印一行。先看的列：

| 列 | 含义 |
|---|---|
| `sim_cycles` | 仿真用的周期数 |
| `hw_cycles` | 同一 launch 在真实 GPU 上的周期数（Nsight Compute 测得） |
| `err_pct` | 两者之差，百分数 |
| `verdict` | `ok` 表示这次运行复现了随发基线 |

这个用例的两个周期数相差约 1 %。其他示例及其预期数字列在 [examples/README.md](../../examples/README.md)。Blackwell 用例用 `--suite blackwell/full_gpu --captures examples/captures/blackwell`。

运行写出的全部内容在 `out/example/sim/<case>/` 下，布局见 [pipeline.md](pipeline.md)。

## 4. 看执行内部

总周期数只是摘要。加 `--ledgers` 还会写出发射日志与等待账本：每条指令何时发射、等了多久、在等什么。

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_einsum_05 \
    --captures examples/captures/hopper --out out/example --ledgers
```

怎么读它们见 [results.md](results.md)。

## 5. 在自己的 GPU 上采集 kernel

采集比仿真要求更多：受支持架构的 NVIDIA GPU、NVBit tracer、做硬件参照的 Nsight Compute，以及套件 kernel 来源的那些库。[capture-tools.md](capture-tools.md) 讲怎么搭好它们。搭好后去掉 `--captures`，用例会先在本地 GPU 上采集：

```bash
cyclescope run --suite hopper/full_gpu --case deepgemm_bf16_dense_02 \
    --out out/captured
```

## 6. 配置

各工具经一个小小的站点文件找到自己的目录。上面的步骤用默认值即可；只有当你的构建目录、tracer 或 Nsight Compute 在别处时才需要它。

把 `configs/site.example.json` 复制为 `configs/site.json`，或用 `CYCLESCOPE_SITE_CONFIG` 指向自己的拷贝。环境变量总是优先于文件。

| 键 | 环境变量 | 设置什么 | 默认 |
|---|---|---|---|
| `artifact_root` | `CYCLESCOPE_ARTIFACT_ROOT` | 工具、capture 与缓存的去处 | `<repo>/artifacts` |
| `build_dir` | `CYCLESCOPE_BUILD_DIR` | 已构建二进制的位置 | `<repo>/build` |
| `tracer` | `CYCLESCOPE_TRACER` | tracer 库，按 SM 架构 | artifact root 之下 |
| `ncu` | `CYCLESCOPE_NCU` | Nsight Compute 二进制 | `PATH` 上的 `ncu` |
| `nvdisasm` | `NVDISASM` | tracer 用的 `nvdisasm` 二进制 | artifact root 之下 |

打印当前生效的值：

```bash
PYTHONPATH=python python3 -m cyclescope.site --show
```

## 接下来去哪

- [pipeline.md](pipeline.md)：逐个阶段，以及一次运行写出的每个文件。
- [results.md](results.md)：读报告与执行记录。
- [suites.md](suites.md)：项目评估所用的 1000 个 kernel。
- [design/README.md](design/README.md)：CycleScope 怎么工作、为什么。
