# 引擎二进制

[English](../en/engine.md)

`cyclescope-engine` 是一次 launch 的逐周期仿真器。输入是一个作业容器（每 cluster 一段，加 launch 级表），输出是一份执行记录：总周期数，以及按要求的逐指令行。本页是接口说明；模型见 [design/engine.md](design/engine.md)。

## 1. 命令行

```bash
cyclescope-engine simulate --job job.bin --out result.bin \
    [--journal j.bin] [--wait-ledger w.bin] [--stall-ledger stall.tsv]
    [--progress]
cyclescope-engine validate --job job.bin   # 结构检查
cyclescope-engine info     --job job.bin   # 段/实例统计
```

stderr 末尾打印 `LOAD=` / `RUN=`（秒）与 `cycles=/issued=/completed=/hard=`。

## 2. 它建模什么

一个简短导览；每项一两行，完整图景在 [design/engine.md](design/engine.md)。

- **编译器排程**：一条指令等待它读的寄存器与控制字指名的记分板，并遵守控制字的 stall 计数。
- **Barrier**：mbarrier、CTA 命名 barrier 与 cluster barrier 是状态机；到达数与传输字节决定一个相位何时完成、等待者何时恢复。
- **异步拷贝与 TMA**：拷贝按 warp 分组；一组先 commit 再等待，一次搬运通过其 mbarrier 报告完成。
- **存储层级**：每 SM 的 L1、共享 L2 与 DRAM，按缓存扇区跟踪，带端口与队列。每次 launch 以全空缓存开始。
- **执行单元**：指令占用它的叶指名的单元，占叶给出的间隔。
- **张量核**：Hopper WGMMA 与 Blackwell UMMA 在每 SM 的张量流水线上排队。
- **线程块分派**：线程块按采集到的放置进入；后续的块随槽位空出而进入。
- **取指**：代码由每 SM 指令缓存加预取供给；launch 开始时为空。
- **逐 warp 队列**（`instruction.warp_queue`）：发射的 STS、LDS、SHFL、LDG、STG 是其 warp 队列的一项，直到串行的释放服务器释放它；队列到深度上限的 warp 不能发射。
- **kernel 边界**：`cycles` 是任一 SM 活跃周期数的最大值——其上 warp 驻留区间 [entry, done) 的并集长度；原先的「末次完成时刻 + envelope」值为遥测 `last_completion_cycle`。
- **架构模块**：Hopper 的 warpgroup 与 WGMMA、Blackwell 的 UMMA、张量存储与 CLC，位于同一接口之后。

## 3. 输出

`simulate --out` 写 `GSIMRES`：总量（`cycles`、`issued`/`completed`、硬约束失败）、逐 SMSP 活跃区间、按桶的等待周期、逐资源等待，以及遥测计数器。`--journal` 与 `--wait-ledger` 再加上逐指令日志与逐 warp 等待账本（[results.md](results.md)，布局在 [formats.md](formats.md) §7）。

遥测行是 `(group, counter, value)`：

| 组 | 计什么 |
|---|---|
| 0 | 引擎总量（发射实例数） |
| 1 | 同步：mbarrier 相位/到达/tx 字节、barrier 代际、轮询、gsync、TMA 完成 |
| 2 | 分派、生命周期与取指：入队、波次、等待周期、取指停驻、逐 CTA 发射跨度 |
| 3 | 架构模块计数器（`wgmma_*`、`umma_*`、`tmem_*` 等） |
| 4 | 存储层级：L1/L2/DRAM 扇区、合并、填充、驱逐、共享服务等待、端口分摊、幻影扇区 |

计数器名是 `cpp/core/simulator.hpp` 的 `SyncTele` / `EngineTele` 枚举（第 4 组顺序固定在 `cpp/core/simulator.cpp`）。

## 4. Stall 账本

`--stall-ledger <path>` 写出一份逐 PC 的 tab 分隔 stall 账本（schema `cyclescope.stall-ledger.v1`）。它只是统计：带该选项的运行产生逐字节相同的 `result.bin` 与日志，没有任何计时判定读账本状态。

```
#schema	cyclescope.stall-ledger.v1
#warps	<变成驻留的 warp 数>
#cycles	<kernel 周期数，与结果报告同值>
residency	<十六进制代码地址>	<operation_class>	<reason>	<warp_cycles>	<events>
transition	<源十六进制代码地址>	<目的十六进制代码地址>	<次数>
```

驻留 warp 被评估发射的每个周期恰好计入一个 `(其下一条指令的代码地址, reason)`；发射周期本身记 `issued`，驻留之前的周期记 `dispatch`；warp 已就绪、只因本 SMSP 的发射槽让给了别的 warp 的周期记 `issue_slot`。reason 是 `<site>[:<detail>]`，在判定处指定：`barrier:<op>[:<detail>][:read]`（等待生产者依赖 barrier）、`unit:<name>` / `unit_outbid:<name>`（`configs/mechanism_tree/common/resources.json` 的单元名）、`semantic:<kind>`、`suspended:<kind>`、`arch:<模块规则>`、`warp_queue:mio` / `warp_queue:lg`（逐 warp 队列）、控制地板 `control:branch` / `control:yield` / `control:bsync`，其余判定处各占一个 site 名（`fetch`、`cta_barrier`、`cluster_barrier`、`control_floor`、`issue_slot`、`exit_branch`、`exit_drain`、`other:<site>` 等）。`transition` 行按 warp 计每对连续发射指令（含顺序 fall-through）。

逐 warp 守恒：

    resident_cycles == (last_issue_cycle - first_resident_cycle + 1)
                       + exit_drain_cycles

## 5. 失败

与管线其余部分相同的两个数据失败类别：作业缺模型需要的事实时报 `missing_fact:<fact> (<where>)`，模型不覆盖时报 `unsupported:<reason> (<where>)`。无法前进的运行以 `unsupported:engine:deadlock` 停止，并转储卡住的 warp 与同步对象状态。

## 6. 给引擎跑基准

```bash
OUT=<dir> [SIZES="S L"] [PROFILE=1] bash scripts/pipeline/engine_perf.sh
```

生成合成作业（S ≈ 25.6 万实例，L ≈ 840 万）、构建 Release 引擎、给 `simulate` 计时，写 `perf.tsv`（墙钟/RSS/周期数/每秒周期数）加可选的 gprof 剖面。

```bash
scripts/pipeline/engine_fixture_digest.sh <engine> <fixtures_dir> <out.tsv>
scripts/pipeline/engine_fixture_digest.sh --compare <a.wrk> <b.wrk>
```

对每个 `*.job` fixture 的结果与日志二进制取哈希；`--compare` 逐表对比两个这样的工作目录（除 WAIT 外每表必须一致）。

## 7. 调试开关

`CYCLESCOPE_DEBUG_*` 环境变量向 stderr 打印诊断；不设即关：

| 变量 | 打印什么 |
|---|---|
| `CYCLESCOPE_DEBUG_INST=<instance>` | 单个实例的就绪检查与写表更新 |
| `CYCLESCOPE_DEBUG_MBARRIER=<seg>:<obj>` | 单个 mbarrier 的全部状态变化 |
| `CYCLESCOPE_DEBUG_CTABAR=<cta>:<barrier>` | 单个 CTA 命名 barrier 的全部状态变化 |
| `CYCLESCOPE_DEBUG_CPASYNC` | cp.async 的发射与等待判定 |
| `CYCLESCOPE_DEBUG_TMA` | TMA 完成计时细节 |
| `CYCLESCOPE_DEBUG_TMEM` | TMEM 分配/释放尝试（Blackwell） |
