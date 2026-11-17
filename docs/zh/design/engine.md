# 引擎：对照机制树重放

[English](../../en/design/engine.md)

采集说明执行了什么；机制树说明硬件如何回应。引擎把每个 warp 采到的指令流对照机制树重放一遍，把每条指令放在硬件允许的最早周期上——如果发射得晚，记下是什么拖住了它。

## 1. 结构模型

"硬件允许什么"需要一个结构。引擎假设一个小结构，当作假设陈述，且只保留还原所必需的部分：

| 抽象 | 实现对象 |
|---|---|
| 带发射槽和 warp 候选表的 SM 子分区 | `cpp/core/simulator.{hpp,cpp}` 中的 SMSP 状态 |
| 编译器的调度：寄存器、记分板、控制字 | 每 warp 的 `reg_ready`/`barrier_free` 表与模板的控制字段 |
| 作为状态机的 barrier | `cpp/core/sync_objects.{hpp,cpp}` |
| 异步操作组 | `cpp/core/async_groups.{hpp,cpp}` |
| 执行单元与共享服务 | 经 `cpp/core/mechanism_runtime.{hpp,cpp}` 解析的资源 |
| 存储层级：每 SM 的 L1、共享 L2、DRAM、store 队列 | `cpp/core/memory_hierarchy.{hpp,cpp}`、`cache_policy.hpp` |
| 代码缓存与预取 | `cpp/core/instruction_fetch.{hpp,cpp}` |
| 线程块分派与 kernel 包络 | `cpp/core/dispatch.{hpp,cpp}`、`lifecycle.hpp` |
| 架构专有机制 | `cpp/arch/architecture_module.hpp` 后面的模块 |

结构里不含数字：其中的每个间隔、带宽、容量和延迟都是一片叶。引擎不计算任何数据值，也不仿真机制以下的任何东西。

## 2. 主循环

引擎按事件推进。`Simulator::run` 维护一个事件队列（`Event::Kind`：`kResultReady`、`kReadBarrierFree`、`kResourceFree`、`kWarpDone`、`kWake`、`kBarrierRelease`、`kDeferredArrive`、`kTransferComplete`、`kDispatch`、`kArch`）和当前周期 `now_`。每一轮：调度所有在 `now_` 到期的事件，然后步进可能发生变化的 SM 子分区——既没有到期重试也没被事件标记的子分区直接跳过——再把 `now_` 推进到最早到期重试和下一个事件两者中较近的那个。时间分辨到周期，但从一次状态变化直接跳到下一次。

`step_smsp` 按轮转顺序遍历子分区的 warp，从上次发射的 warp 之后开始（若那条指令没有 yield 则从它本身再来），每周期发射的条数以 `scheduler.policy` 叶的 `issue_slots` 字段为限。一个 warp 被判定阻塞时扫描即停，除非 `try_next_on_block` 叶另有规定。

## 3. 这条指令能发射吗

就绪检查按固定顺序进行（`try_issue` 与 `step_smsp` 中的扫描）：

1. **dispatch**——warp 的 CTA 还没被放置，或者它的进入周期还没到；
2. **fetch**——这条指令的代码行不在代码缓存里；
3. **挂起**——warp 被泊在一个已知的释放周期之前（`blocked_until`）；
4. **集体等待**——延迟的 mbarrier 等待、CTA barrier、cluster barrier；
5. **语义**——这条指令对 barrier 或组做的事还没满足（轮询、async 组等待、架构专有的语义）；
6. **资源**——它的执行单元在忙、端口或发射间隔未到、操作数收集器被占、架构的访存队列持有适用；
7. **寄存器**——源寄存器、记分板或控制字下限还没就绪；
8. **架构模块**——架构自己的就绪判定。

第一个失败的检查就是这次等待的原因，在下一个判定之前计入七个桶之一：

| 桶 | 含义 |
|---|---|
| `kDispatch` | 等 CTA 被分派、warp 进入 |
| `kFetch` | 等取指 |
| `kCollective` | 等 barrier 释放或挂起唤醒 |
| `kSemantic` | 等语义门（一次轮询、一个 async 组、一条架构规则） |
| `kResource` | 等在忙的单元、端口、队列或发射间隔 |
| `kRegister` | 等寄存器值或编译器的调度 |
| `kScheduler` | 不归因于以上任何检查：warp 第一次检查失败前的初始原因 |

## 4. 编译器的调度

静态事实解决第一种停顿。指令等它读取的寄存器和它的控制字指明的记分板；它也遵守控制字的 stall 计数，而 yield 位只轮转候选顺序。设置某个寄存器或记分板的指令在它的叶给出的延迟之后交付结果，这时寄存器就绪、记分板清除。引擎不对调度做任何推断：它照编译器写的执行。

## 5. 其他 warp

运行时事实解决第二种停顿。

- **mbarrier**（`MbarrierTable`）：对象跟踪期待到达数、待到达数、未完成的事务字节数和相位；采集到的操作数指明是哪个对象、多少到达和字节、哪个奇偶——代际本身在这里求解。到达数与字节数都达到期待值时相位完成，等待者在实测的释放延迟之后恢复。
- **CTA 命名 barrier 与 cluster barrier**：按同样方式计代际，分别在 CTA 内和 cluster 内（`CtaBarrierTable`、`ClusterBarrierTable`）。
- **异步组**（`AsyncGroups`）：bulk、`cp.async` 与 warpgroup 三条通道；commit 关闭当前打开的组，wait 至多允许 N 个更新的组仍待完成。
- **借全局存储做的同步**（`global_sync.{hpp,cpp}`）：一次读与产生 trace 观测到的那个值的写配对，那次写完成并可见之后，这次读就绪。

逻辑状态是精确的——按 PTX 语义更新；只有时长来自叶。

## 6. 硬件容量

机制事实解决第三种停顿。

- **存储层级**：每次采到的访问事先拆成了扇区。模型跟踪哪些扇区驻留在每 SM 的 L1 与共享的分区 L2（含读侧副本与脏行写回）、哪些填充还在途中、每 SM 端口与 DRAM 队列；对同一未命中的读会合并（`memory.l2_merge`）。每次 launch 以全空缓存开始——参照条件是冷缓存。
- **执行单元与共享服务**：指令占用它的叶指名的资源，占叶给出的间隔；全卡服务（L2 与 DRAM 带宽、分派）由所有 SM 共享。
- **共享存储与异步搬运**：按 SMSP 与按 SM 的共享服务接受率，TMA 与 bulk copy 的服务开始与完成（`async_groups`、`memory_hierarchy`）。
- **分派与生命周期**（`dispatch`、`lifecycle`）：第一波放置来自采到的 `PLCE` 行，排队 CTA 在槽位释放时进入，串行进入间隔，以及每次 launch 的包络。
- **取指**（`instruction_fetch`）：每 SM 的代码缓存，带顺序预取与 launch 预取；代码行不在就停在 fetch 桶。

## 7. 架构模块

核心不按设备分支。所有 Hopper 或 Blackwell 专有的东西都在一个接口后面（`cpp/arch/architecture_module.hpp`），每次仿真由 `cpp/arch/registry.cpp` 按作业的 device 实例化：

```cpp
virtual bool     instance_ready(Simulator&, WarpState&,
                     const InstructionInstance&, const InstanceSemantics&);
virtual void     on_instance_issue(Simulator&, WarpState&, …, finish);
virtual void     on_event(Simulator&, const Event&);     // incl. Event::kArch
virtual uint64_t memory_queue_retry(sm, smsp, now);
virtual IssueSteal issue_steal(const Simulator&, const WarpState&);
virtual bool     coresident_dependent_penalty(…);
virtual void     on_load(Simulator&);  on_warp_exit(…);  collect_tele(…);
```

Hopper 模块加入 warpgroup 集合操作与 WGMMA 组 FIFO、累加器交接、寄存器堆发射窃用和 proxy fence。Blackwell 模块加入 UMMA 发射与 commit、张量存储预算与 load/store、CLC 查询和异步 store。见 [arch-semantics.md](arch-semantics.md)。

## 8. 输出

`cyclescope-engine simulate --job job.bin --out result.bin` 写出执行记录：

- **结果**（`GSIMRES`）：总周期数、发射与完成的指令数、按桶与按资源的停顿周期、计数器；
- **发射日志**（`GSIMJNL`，`--journal`）：每条已发射指令一行 `JournalRow`——warp、SMSP、实例、发射周期、等待原因与周期、所等待的对象；
- **等待账本**（`GSIMWLG`，`--wait-ledger`）：按 warp 与原因计的等待周期。

布局见 [../formats.md](../formats.md)；计数器与遥测的含义见 [../engine.md](../engine.md)。

## 9. 失败

若没有任何 warp 能前进——没有到期重试、没有待处理事件——引擎转储阻塞状态并以 `unsupported:engine:deadlock` 停止。`Simulator::Options::max_cycles` 可以限制一次运行；越过它就停止并计一次硬失败。命令行不暴露该选项。

## 10. 代码位置

| 文件 | 作用 |
|---|---|
| `cpp/core/simulator.{hpp,cpp}` | 主循环、事件队列、SMSP 扫描、就绪顺序、等待桶、发射日志 |
| `cpp/core/mechanism_runtime.{hpp,cpp}` | 把 USES 行解析成叶字段；engine 轴；ns→cycles；选择缓存 |
| `cpp/core/generated/parameter_schema.hpp` | 生成的按模式字段 id |
| `cpp/core/sync_objects.{hpp,cpp}` | mbarrier、CTA barrier、cluster barrier 状态机 |
| `cpp/core/async_groups.{hpp,cpp}` | bulk / cp.async / warpgroup 组通道 |
| `cpp/core/global_sync.{hpp,cpp}` | 全局存储同步的 acquire 计划与配对 |
| `cpp/core/memory_hierarchy.{hpp,cpp}` | L1 / L2 / DRAM、端口、在途填充、读合并 |
| `cpp/core/cache_policy.hpp` | 加权全局 LRU 与 cache-hint 行为 |
| `cpp/core/instruction_fetch.{hpp,cpp}` | 代码缓存与预取 |
| `cpp/core/dispatch.{hpp,cpp}` | CTA 分派、占用、进入间隔 |
| `cpp/core/lifecycle.hpp` | 每次 launch 的包络 |
| `cpp/core/job_format.hpp`、`job_reader.{hpp,cpp}` | 二进制作业记录与合并容器读方 |
| `cpp/core/result_writer.{hpp,cpp}` | `GSIMRES` / `GSIMJNL` / `GSIMWLG` 写出 |
| `cpp/core/semantics_kinds.hpp` | 引擎据以分派的 `SEMA.kind` 枚举 |
| `cpp/arch/architecture_module.hpp`、`registry.cpp` | 架构接口与模块选择 |
| `cpp/arch/hopper/` | `module`、`wgmma`、`proxy_fence`、`capture_rules`、`convert_rules` |
| `cpp/arch/blackwell/` | `module`、`umma`、`tmem`、`clc`、`stas`、`capture_rules`、`convert_rules` |
| `cpp/main.cpp` | `cyclescope-engine` 子命令 `simulate`、`validate`、`info`、`dump` |
| `cpp/tests/` | CTest 夹具：simulator、存储层级、同步、取指、架构模块、作业读方 |
