# 管线，逐阶段

[English](../en/pipeline.md)

一个用例流经六个阶段。`cyclescope run` 为单个用例驱动全部阶段；每个阶段也是独立命令。两个阶段需要 GPU（采集与硬件参照）；其后的都在 CPU 上，所以任何机器都能回放已有 capture。

```
capture ──► reference ──► convert ──► build ──► simulate ──► report
（tracer，   （ncu，        （事实 +      (job.bin)  （结果 +     （与基线
 GPU）        GPU）          段）                    账本）       的一行）
```

- **capture** 在 NVBit tracer 下运行用例，记录每个 warp 执行了什么（运行时事实）以及 SASS（静态事实）。
- **reference** 用 Nsight Compute 在不带 tracer 的情况下给同一 launch 计时。它是报告拿来比较的硬件数字。
- **convert** 把 capture 变成逐 cluster 的事实与段——不依赖时间就能定下的全部内容。
- **build** 把各段与 launch 级表打成一个作业容器。
- **simulate** 让 C++ 引擎对着机制树仿真，写出执行记录。
- **report** 每个用例打印一行，并与随发基线比较。

## 1. `cyclescope run`

```bash
cyclescope run --suite hopper/full_gpu --case <case> --out <dir> \
    [--captures <root>] [--ledgers] [--keep] [--jobs N]
```

它从套件表解析用例、按序运行各阶段，遇到第一个失败即停。给了 `--captures <root>` 就跳过 GPU 阶段：用例 capture 取 `<root>/<case>`，参照取 `<root>/reference/<case>`（若存在）。

| 标志 | 作用 |
|---|---|
| `--suite`、`--case`、`--out` | 跑哪个套件用例、输出树放哪 |
| `--captures` | 复用已有 capture 根，不再采集 |
| `--reference-tsv` | 用这张参照表代替 capture 自带的 |
| `--skip-reference` | 不跑 ncu 阶段（没有 `hw_cycles` 列） |
| `--tree` | 绑定的机制树（默认：该套件生成的树） |
| `--engine` | 引擎二进制（默认：`PATH` 上的 `cyclescope-engine`） |
| `--builder` | `cpp`（默认）或 `python` 段构建器 |
| `--jobs` | 转换分片的并行数（默认：分片数，封顶 CPU 数） |
| `--shard-records` | 超过此记录数转换即分片（默认 150 万） |
| `--shard-max` | 分片数硬上限（默认 16） |
| `--no-placement` | 转换时不带采集到的 CTA→SM 放置 |
| `--keep` | 保留中间的 `facts/`、`segments/`、`job.bin` |
| `--ledgers` | 同时写出发射日志与等待账本 |
| `--repeats` | 参照的 ncu 重复次数（默认 3） |
| `--timeout` | 采集/参照子进程的超时秒数（默认 5400） |
| `--tracer` | tracer 库路径（默认：站点配置） |
| `--scratch` | 在此目录做转换工作，结果拷回 |

## 2. 手工逐阶段运行

```bash
cyclescope capture    --suite <suite> --case <case> --out <dir>
cyclescope reference  --suite <suite> --cases <case> --out <dir> [--repeats N]
cyclescope convert    --capture <dir> --arch <arch> --out <facts>
cyclescope convert    --facts <facts> --arch <arch> --out <dir>
cyclescope build      <dir> --out job.bin
cyclescope-engine simulate --job job.bin --out result.bin \
    [--journal j.bin] [--wait-ledger w.bin] [--progress]
cyclescope report     <out>
```

`capture` 还接受 `--mode {trace,placement,both}`、`--ncu-repeats`、`--timeout`、`--dry-run` 与 `--reference`（采集参照运行所计时的那个 launch）。`reference` 接受 `--captures`（给每个 capture 所采的 launch 计时）、`--metrics`（额外 ncu 指标，存为 `ncu_metrics.csv`）与 `--source-stalls`（SASS 级停驻采样 pass，存为 `ncu_source.csv`）。`convert` 接受 `--shard k/N`、`--split-shards N`、`--builder`、`--tree`、`--cache`、`--no-cache`、`--launch`、`--clusters`、`--reference`、`--placement`、`--reference-sm-clock-khz` 与 `--reference-cache-control`。

辅助命令，各一行：

- `cyclescope split <job.bin> --out <dir>`——把容器拆回段目录。
- `cyclescope relink <seg|job.bin> --tree <t.json>`——把机制程序表重链到另一棵生成的树上，不重新转换。
- `cyclescope tree build [--emit-cpp] | validate [--strict] | compile | diff | show`——机制树工具（[design/mechanism-tree.md](design/mechanism-tree.md)）。
- `cyclescope-engine validate|info|dump --job <file>`——不仿真地检查一个作业。

## 3. 一次运行写什么

```
<out>/
  captures/<case>/            capture 目录（run.json、launches.jsonl、
                              functions/、tensor_maps.jsonl、
                              trace-<launch>.bin.zst、placement/ …）
  captures/reference/<case>/  硬件参照（ncu 重复、reference.tsv）
  sim/<case>/
    status.json               最终状态（下表）
    stages.tsv                每阶段一行（第 6 节）
    convert.log segments.log build.log sim.log
    facts/ launch.json        逐 cluster 事实（--keep 时保留）
    segments/                 GSIMSEG 段（--keep 时保留）
    job.bin                   GSIMJOB 容器（--keep 时保留）
    out-<builder>/
      result.bin              执行记录总量（GSIMRES）
      launch.json             launch 级作业文档
      journal.bin             发射日志   （--ledgers 时）
      wait.bin                等待账本   （--ledgers 时）
```

`status.json` 是 `cyclescope report` 读的东西。字段：

| 字段 | 含义 |
|---|---|
| `suite`、`case` | 这是哪个用例 |
| `stage` | 运行停在哪一阶段（`capture` `reference` `convert` `segments` `build` `sim`）或 `done` |
| `class` | `ok`，或某个失败类别（第 5 节） |
| `records` | trace 中的动态指令记录数 |
| `dropped_polls` | 折叠成等待的轮询观测数 |
| `missing`、`unsupported` | 按缺事实 / unsupported 键的计数 |
| `issued`、`completed` | 引擎发射 / 退出的指令实例数 |
| `cycles` | 仿真周期数 |
| `hard` | 引擎计到的硬约束失败数 |
| `walls` | 每阶段墙钟秒数（`convert`、`segments`、`build`、`sim`、`total`） |
| `rss_kb` | 每阶段内存峰值，KB |
| `hw_cycles` | 硬件参照周期数（有参照时） |
| `reference_cache_control` | 参照采集时的缓存条件 |
| `placement_digest` | 采集的 CTA→SM 放置的 SHA-256 前 16 个十六进制位——报告拿它与基线比较 |
| `shards` | 跑了几个转换分片 |

## 4. 大 capture：分片

一个 capture 的记录数可能装不进一次转换。此时转换器按分片工作：`--split-shards N` 按 cluster 切开被采的 launch，每片跑 `convert --capture … --shard k/N` 加 `convert --facts …`，分片并行（`--jobs`），事实与段表带一致性检查地合并。`cyclescope run` 自己规划分片：记录数超过 `--shard-records`（默认 150 万）即触发，最多 `--shard-max` 片（默认 16）。`stages.tsv` 每片一行；`status.json` 在 `shards` 里报分片数。

## 5. 运行失败时

数据问题只有两个失败类别：

- `missing_fact:<fact> (<where>)`——管线需要的事实不在 capture 里。不用默认值顶替。
- `unsupported:<reason> (<where>)`——事实在，但模型不覆盖它。引擎死锁报为 `unsupported:engine:deadlock`。

`status.json.class` 还用 `crash`（某阶段非零退出）与 `issued_ne_completed`（引擎跑完但并非每个实例都退出）。先看阶段日志：`sim/<case>/` 下的 `convert.log`、`segments.log`、`build.log`、`sim.log`。

接下来看[代理手册](agents/README.md)：分诊表按阶段和类别把症状对应到采集失败、转换失败、引擎失败、缺叶值、缺机制各篇。

## 6. 给一次运行计时

各阶段用时在 `sim/<case>/stages.tsv`，每阶段一行：

```
<stage>\t<wall_s>\t<max_rss_kb>\t<rc>\t<cpu_s>
```

`cpu_s` 是该阶段进程树的用户+系统时间；分片阶段另有每片一行，所以加总可能超过墙钟。`status.json` 在 `walls` / `rss_kb` 里镜像同样的数。

比较仿真器速度时，每个用例计入的是 trace 文件落盘到结果写出之间的 CPU 侧工作：trace 解析、事实提取、作业构建、仿真及其间的 I/O。插桩 GPU 运行、ncu 参照、排队等待与报告聚合都不算。把成功用例一一配对，报墙钟与 CPU 时间的比值，并说明哪侧用了用例内并行。
