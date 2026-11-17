# 在自己的 GPU 上采集：工具与设置

[English](../en/capture-tools.md)

回放随发 capture 只需要构建。*采集*一个 kernel 需要更多：NVBit tracer、做硬件参照的 Nsight Compute、CUDA 工具链，以及套件 kernel 来源的第三方源码。这些都不随仓库 vendored——每个外部件都按名字登记在 `configs/capture_tools.json` 里，放在 `$CYCLESCOPE_ARTIFACT_ROOT/tools/<name>`。

## 1. 在新机器上搭建

`scripts/provision/provision_tools.py` 拉取、构建并检查每个登记项：

```bash
export CYCLESCOPE_ARTIFACT_ROOT=<工具存放处>
python3 scripts/provision/provision_tools.py plan          # 步骤，按依赖序
python3 scripts/provision/provision_tools.py fetch --all   # 或：fetch <name>...
python3 scripts/provision/provision_tools.py build <name>  # 在其 toolchain 环境内
python3 scripts/provision/provision_tools.py verify --all
python3 scripts/provision/provision_tools.py bundle --out DIR   # 导出给离线机器
```

顺序：设 artifact root，看 `plan`，`fetch` 全部可下载项，在它登记的 toolchain 环境里跑每个 `build` 项（`cd <clone> && scripts/provision/provision_tools.py build <name>`），放好手工下载件，然后 `verify --all`。没有网络的机器用 `bundle` 导出干净结果。

注意：

- 默认 PyPI 索引拉不到时设 `CYCLESCOPE_PIP_INDEX_URL`；`CYCLESCOPE_TMP` 控制拉取所用的临时目录。
- Nsight Compute 是 `manual` 项——provisioner 下载不了；把它放到 `$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute`，让 `verify` 检查。
- `build` 项在其 toolchain 环境内运行；在宿主机上构建得不到预期产物。
- 构建若以 root 跑，产物归 root；`CYCLESCOPE_OWNER` 让 provisioner 把它们 chown 回来。

## 2. 登记项的种类

`configs/capture_tools.json` 每项有一个 `kind`，决定它怎么产生（各项字段：见文件）：

| 种类 | 是什么 |
|---|---|
| `git` | 钉在固定修订的上游检出，可选补丁与子模块拉取 |
| `pypi_wheels` | 按锁定 requirements 清单安装的 Python wheels |
| `download` | 按校验和拉取并解压的归档或文件集；`manual: true` 表示人工下载 |
| `build` | 登记构建脚本在 toolchain 内跑出的产物 |
| `composite` | 把其他项的子树用符号链接拼起来 |
| `toolchain` | 一组 `build` 项运行所需的环境 |
| `overlay` | 盖在另一项树上的文件 |

## 3. 一棵树怎么检查

每项记录一个 `tree_sha256`：所有文件与符号链接按路径排序，各与其内容（或链接目标）一起哈希，整张清单再哈希。`verify` 重新计算——只有相符才算过，残缺或损坏的 provision 混不过去。`scripts/provision/tree_sha256.py` 手工重算同一摘要。

## 4. 采集时要知道的

- **JIT 缓存按用例重建。** 它们是输出不是输入——每个用例拿到一个空的 scratch 缓存。冷 JIT 在某些用例上要几分钟，所以首次采集可能需要比默认 5400 秒更大的 `--timeout`。
- **Nsight Compute 因架构而异。** H100 的环境不带 ncu——把 `CYCLESCOPE_NCU` 指向 provision 好的 `$CYCLESCOPE_ARTIFACT_ROOT/tools/nsight-compute`。B300 用环境自带的 `ncu`。每份参照实际用的版本记在用例的 `timing.json` 里；确切版本号见 [environment.md](environment.md)。
- **超时停掉整个进程组。** 采集与参照的子进程在独立进程组里跑，超时不会给后续运行留下挂着的 profiler。

采集或硬件参照失败时，见 [agents/capture-failures.md](agents/capture-failures.md)。要采集套件之外的 kernel 或库，见 [agents/new-kernel.md](agents/new-kernel.md) 与 [agents/new-library.md](agents/new-library.md)。

## 5. 比较 capture 与工具输出

两个脚本，各一行：

- `scripts/provision/compare_tool_outputs.py`——把某个 provisioned 项或构建产物与参照树对比。
- `scripts/pipeline/compare_captures.py`——对比两份 capture 的静态事实（launch 清单、functions、SASS、tensor map）。
