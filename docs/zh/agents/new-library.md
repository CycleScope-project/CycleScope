# 采集新库

[English](../../en/agents/new-library.md)

把仓库从未采集过的库/框架拉起来，顺序：可构建 → 可复现 →
case 规格 → 首次采集。

## 1. 让资产可复现

所有外部资产经 `configs/capture_tools.json` 进入（条目 kind：
`git`、`pypi_wheels`、`download`、`build`、`composite`；完整性由
`tree_sha256` 保证）：

1. 加条目：来源（`url`/`rev`、wheels 或下载 URL）、`patches`
   （仓库内补丁文件）、`toolchain`（`build` 条目用
   `h100`/`b300` —— 构建脚本在该工具链环境里跑）、`outputs`、
   `purpose`、`cases`。
2. 供应：`python3 scripts/provision/provision_tools.py fetch
   <name>`（`build <name>` 在工具链环境内），然后 `… verify
   <name>`（树哈希须等于 `tree_sha256`；条目合法变更时用
   `scripts/provision/tree_sha256.py` 重新钉）。见
   [../capture-tools.md](../capture-tools.md)。

## 2. 让它可运行

- Hopper：case 是 `cases.tsv` 里一行 argv/env —— 独立二进制，
  或 `suites/hopper/runners/` 下的小 harness 脚本（`env_profile:
  far` 如何用 `environment_far.sh` 可作参考）。
- Blackwell：`suites/blackwell/runners/<family>/` 下的 runner 模
  块 + `runners/common/capture_adapters.py` 的 family 适配器 +
  `cases_<family>.json` 规格；新 family 需要适配器代码（未在发
  布树上实操）。

## 3. JIT 编译型库

- 缓存/工作区写在 `<CAPTURE>`（per-case 临时目录），绝不用共享
  路径 —— 并行 case 不能共享 JIT 缓存。
- 冷 JIT 可能要数分钟：相应调大 `cyclescope capture --timeout`
  （默认 1800 s）与 `reference --timeout`。
- ncu census prepass 在被跟踪运行前先热缓存，参照与采集走同一
  代码路径；triton 系 case 用 prepass 钉住 autotune 配置
  （`python/cyclescope/capture/driver.py` 的 `prepare_case`）。

## 4. launch 选择陷阱

- launch 很多：对照 ncu census（`ncu_launches.jsonl`）选
  `kernel_regex` + `occurrence`，不要凭目测顺序。
- 预热 launch：要计入 —— `occurrence` 是 census 里第 N 个匹配，
  包含预热。
- 主 kernel vs 辅助 kernel：正则必须只匹配目标（mangled 名）；
  过宽会静默选中辅助 kernel。
- 每次运行 kernel 不同（JIT 名后缀、autotune 变体）：先钉配置
  （prepass/autotune 文件），否则 census 与被跟踪运行看到不同
  kernel。

## 5. 常见首批失败

| 首个失败 | 去 |
|---|---|
| `no traced launch`、`selection_failed`、census 未命中 | [capture-failures.md](capture-failures.md) §2 |
| 新 opcode/操作数形式上的 `missing_fact:*`/`unsupported:*` | [convert-failures.md](convert-failures.md) |
| 新同步模式上的 `unsupported:engine:*`/死锁 | [engine-failures.md](engine-failures.md) |
| 时间不对、机制存在但叶缺失/占位 | [missing-leaf.md](missing-leaf.md) |
| 完全没有机制规则的行为 | [missing-mechanism.md](missing-mechanism.md) |

## 6. 可复现检查单

- [ ] 资产在 `configs/capture_tools.json` 中且 `tree_sha256` 已
      钉，`provision_tools.py verify <name>` 通过
- [ ] case 规格已入仓库（`cases.tsv` 行 / family JSON + 适配
      器），非手敲命令行
- [ ] JIT 缓存在 `<CAPTURE>` 下；冷 JIT 在 `--timeout` 内
- [ ] `cyclescope capture --dry-run` 显示预期 argv/env/cwd 与
      `launch_selection`
- [ ] `trace_check` 通过且 `ncu_launches.jsonl` 含选中的
      launch
- [ ] census 与被跟踪运行选中同一 kernel（必要时钉
      autotune/JIT 配置）
