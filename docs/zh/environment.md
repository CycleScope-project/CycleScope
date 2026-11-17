# 参考环境

[English](../en/environment.md)

*参考环境*是 CycleScope 随发数字产生所用的机器：跑 capture 与硬件参照的 GPU，以及构建工具的主机。照这里列出的版本搭环境是复现基线最稳的路；其他配置未经验证。完整记录——含包清单（`pip_freeze`）——在 `configs/capture_environment.json`。非 pip 字段如下。

## 主机（构建 / 转换 / 仿真）

| 字段 | 值 |
|---|---|
| os | Ubuntu 22.04 |
| g++ | 11.3.0 |
| cmake | 3.22.1 |
| python | 3.10.6 |
| zstd | 1.4.8 |
| 库 | OpenSSL libcrypto 开发文件 |

## h100（采集 + 硬件参照）

| 字段 | 值 |
|---|---|
| gpu | NVIDIA H100 |
| driver | 570.124.06 |
| os | Ubuntu 22.04 |
| cuda | 12.8 runtime，nvcc release 12.9 (V12.9.86) |
| g++ | 11.4.0 |
| python | 3.12.13 |
| ncu | Nsight Compute 2025.1.0.0 (build 35237751)，单独安装（`$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute`）；环境本身不带 |
| nvbit | 1.8 |
| 关键包 | torch 2.10.0+cu129、triton 3.6.0（triton 用例用 provision 的 3.7.0 wheels）、numpy 2.2.6、nvidia-cutlass-dsl 4.4.2、cuda-python 12.9.4、flashinfer-python 0.6.6（far profile 用 provision 的 0.6.17 site） |
| capture_env | `CYCLESCOPE_CUDA_STACK_SIZE=65536` |

## b300（采集 + 硬件参照）

| 字段 | 值 |
|---|---|
| gpu | NVIDIA B300 SXM6 AC (SM103) |
| driver | 580.159.03 |
| os | Ubuntu 24.04 |
| cuda | 13.2，nvcc release 13.2 (V13.2.78) |
| g++ | 13.3.0 |
| python | 3.12.3 |
| ncu | Nsight Compute 2026.1.1.0 (build 37634170)，来自环境 |
| nvbit | 1.8 |
| 关键包 | torch 2.12.0+cu132、triton 3.7.0、numpy 1.26.4、nvidia-cutlass-dsl 4.5.2、cuda-python 13.4.1 |
| capture_env | `CYCLESCOPE_CUDA_STACK_SIZE=65536`、`FLASHINFER_NVCC_THREADS=4`、`MAX_JOBS=4` |

## 参考卡

`configs/hardware/<device>/` 存放管线读取的逐卡事实：

- `device.json`——设备名、`sm_arch`（`sm_90a` / `sm_103a`）、B300 `tmem_columns`。
- `sm_topology.json`——每 SM 一行（`sm`、`placement_domain`、`l2_partition`、`tpc_pair`、`gpc`）：h100 132 个 SM，b300 148 个（六个 b300 SM 无 GPC 指派——`4294967295` 哨兵）。
- `preinitialized_mbarriers.json`（仅 b300）——按 cta_group 展开的硬件保留 mbarrier。

`launch.dispatch` 的 `slow_gpc_mask` 叶声明这两张卡上哪个 GPC 更慢——这是关于卡的声明事实，不是拟合值。

在这些环境上检查过什么、没检查什么：[release-validation.md](release-validation.md)。
