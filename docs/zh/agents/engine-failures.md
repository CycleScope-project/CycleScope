# 引擎失败

[English](../../en/agents/engine-failures.md)

模拟器自身报告失败。终态类在 `sim/<case>/status.json`（`sim` 阶
段）；抛出的字符串与停驻 warp 报告在 `sim.log` 与
`out-<builder>/` 下。以下类是 `cpp/core/`、`cpp/arch/<arch>/`
模块与 `cpp/main.cpp` 中抛出的 `EngineError` 字符串。转换期类
（`ConversionError`、Python `convert/`）见
[convert-failures.md](convert-failures.md)。

## 1. 引擎抛出的 `unsupported:engine:*`

| 类（文件） | 含义 / 首查 |
|---|---|
| `unsupported:engine:deadlock`（`simulator.cpp`） | 无 warp 可推进；stderr 先打印 `deadlock:` 报告 —— 见 §4 |
| `unsupported:engine:discrete_miss`（`mechanism_runtime.cpp`） | 离散叶在无有行的轴值上求值 —— 见 [missing-leaf.md](missing-leaf.md) |
| `unsupported:engine:discrete_axis_id`（`mechanism_runtime.cpp`） | 离散轴选择行解析出轴域外的 id —— 树片段与 `axes.json` 不一致 |
| `unsupported:engine:unknown_axis`（`mechanism_runtime.cpp`） | 机制引用了树中不存在的轴名 |
| `unsupported:engine:parm_model`（`mechanism_runtime.cpp`） | 叶的参数模型种类运行时无法识别 —— schema/树不一致 |
| `unsupported:engine:no_reference_clock`（`mechanism_runtime.cpp`、`simulator.cpp`） | 字段需 ns→拍换算但作业无 `reference_sm_clock_khz` |
| `unsupported:engine:use_index`（`mechanism_runtime.cpp`） | USES 索引引用了作业没有的 capture-facts 块 |
| `unsupported:engine:resource_state`（`simulator.cpp`） | 资源的在途完成无法腾出槽位 —— 资源模型走入死端 |
| `unsupported:engine:fetch_line_bytes`（`simulator.cpp`） | fetch 机制开启但 `line_bytes == 0` —— 叶值缺失/为零 |
| `unsupported:engine:operand_read_collector`（`simulator.cpp`） | 运行时不认识的 operand-read collector id |
| `unsupported:engine:wait_block_offset`（`simulator.cpp`） | 延迟的 mbarrier 等待将阻塞越过 `end_instance` —— 段/事实不一致 |
| `unsupported:engine:mbarrier_uninitialized`（`sync_objects.cpp`） | 对从未初始化的 mbarrier 对象 arrive/wait |
| `unsupported:engine:mbarrier_reinit`（`sync_objects.cpp`） | 对仍有未决 arrive/tx/waiter 的 mbarrier 重复初始化 |
| `unsupported:engine:mbarrier_overrun`（`sync_objects.cpp`） | arrive 数超过该相位预期计数 |
| `unsupported:engine:cta_barrier_overrun`（`sync_objects.cpp`） | CTA barrier arrive 超过该代计数 |
| `unsupported:engine:inter_cta_global_sync:<detail>`（`global_sync.cpp`） | 语义冲突或无法解析的全局同步轮询模式 |
| `unsupported:l2_merge_domain` / `unsupported:l2_merge_domain_gpc`（`simulator.cpp`、`memory_hierarchy.hpp`） | 算出的 L2 merge-domain id 超出可表示范围 —— 拓扑/配置不一致 |
| `unsupported:umma_kind`（`arch/blackwell/umma.cpp`） | 引擎未实现的 UMMA issue kind 位（同一字符串也由 `convert_rules.cpp` 在转换期抛出） |

## 2. 引擎抛出的 `missing_fact:*`

拓扑/launch 形状：`sm_topology`、`sm_topology.gpc`、`sm_count`、
`launch_block_dims`、`occupancy_blocks_per_sm` —— 全在
`simulator.cpp`；作业缺全卡运行所需的 placement/拓扑事实。
程序/服务解析：`instruction.service`（`simulator.cpp` —— opcode
在树中无服务类）；`missing_fact:mechanism_leaf`、
`missing_fact:axis:<name>`、`missing_fact:axis:tensor_k`
（`mechanism_runtime.cpp`、`arch/blackwell/umma.cpp` —— 机制需要
的叶或轴不在作业中）。
同步对象：`mbarrier_object_table`（`simulator.cpp`，对象 id 无
法经 MBOB 表解析为 rank/地址）、`mbarrier_init_count`、
`mbarrier_arrive_count`（`sync_objects.cpp`、`simulator.cpp`，
arrive/wait 缺 init/预期计数事实）、
`global_counter_initial_value`（`global_sync.cpp`，全局同步计
数器无记录的初值）。
数据搬运：`tma_transfer_bytes`（`simulator.cpp`，TMA 完成不带
字节数）、`tmem_columns`（`arch/blackwell/tmem.cpp`，tmem 分配
无列数）。

## 3. 等待桶

journal / wait ledger 把每 warp 的停顿时拍记入
`WarpState::last_wait` / `block_bucket`，枚举在
`cpp/core/simulator.hpp`：

| 桶 | 何时记账（设置处） |
|---|---|
| `kDispatch` | warp 被分派节流（`simulator.cpp` 分派路径） |
| `kFetch` | 等待取指行 |
| `kCollective` | 集合同步：mbarrier 相位等待、cp.async/TMA 组等待、`nanosleep` 唤醒、barrier 等待 |
| `kSemantic` | 语义等待：全局同步轮询窗口、scoreboard/语义依赖 |
| `kResource` | 执行单元或共享资源忙（unit waiter、寄存器堆端口、资源背压） |
| `kRegister` | 寄存器依赖（scoreboard/寄存器就绪重试） |
| `kScheduler` | 默认/调度 —— 未记录具体等待 |

ledger 按 warp 按停顿记录；journal（`--journal`）记指令级事件。
遥测计数器（`simulator.cpp` 的 `tele_[]`，如
`kTeleGsyncWaitCycles`、`kTeleNanosleeps`）聚合具名等待种类 —
— 全表见 `simulator.cpp`。

## 4. 调试开关（`cpp/` 中 `getenv`）

`CYCLESCOPE_DEBUG_TMEM`（Blackwell tmem 分配）、
`CYCLESCOPE_DEBUG_TMA`、`CYCLESCOPE_DEBUG_CTABAR`、
`CYCLESCOPE_DEBUG_MBARRIER`、`CYCLESCOPE_DEBUG_CPASYNC`、
`CYCLESCOPE_DEBUG_INST`（参数 = 实例索引；开启该实例诊断）。加
在 `cyclescope-engine simulate` 上。

`cyclescope-engine` 子命令：`simulate --job <j> --out <r>
[--journal f] [--wait-ledger f]`、`info --job`、`dump --job`、
`validate --job`。`cyclescope run --ledgers --keep` 把 journal
与 wait ledger 留在 `out-<builder>/`。`scripts/pipeline/
compare_results.py` 对两个 `result.bin` 求差。

## 5. 死锁排查步骤

1. `unsupported:engine:deadlock` 之前 stderr 会先打 `deadlock:`
   报告（停驻 warp、指令、等待桶）—— 在 `sim.log` 里读；不够
   再加 `--journal`/`--wait-ledger` 重跑。
2. 找 warp 等的对象（mbarrier / CTA barrier / cluster barrier /
   async 组 / 全局同步轮询 / 资源）。
3. 查对象事实：`facts/launch.json` 与对应段行中的
   `mbarrier_init_count`、`mbarrier_arrive_count`；全局同步轮
   询查 `global_counter_initial_value` 事实与
   `inter_cta_global_sync` 记录。
4. 判定：事实从未记录 → 采集缺口（`tools/tracer/`）；记录了但
   段行错/缺 → 转换规则（`python/cyclescope/arch/<arch>/
   rules.py` + `cpp/arch/<arch>/convert_rules.cpp`）；事实齐且
   合理 → 引擎规则（共享对象在 `cpp/core/`，arch 模块如 tmem 在
   `cpp/arch/<arch>/`）。
