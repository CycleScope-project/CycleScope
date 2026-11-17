# 设计概览

[English](../../en/design/README.md)

CycleScope 回答一个问题：这个 kernel 在这块 GPU 上运行时，逐周期发生了什么？本页讲这个想法，其余设计注记按一次 launch 流经各部分的顺序逐一展开。

## 1. 问题

关于一块 GPU，可以问两个不同的问题。

- 硬件架构师问：如果 GPU 这样造，**会发生什么**。机器是假想的，所以它的行为必须建模，参数必须配置或校准。
- kernel 开发者问：这个 kernel 在一块存在的 GPU 上**发生了什么**。执行已经发生，所以可以采集；机器存在，所以它的行为可以测量。

CycleScope 为第二个问题而建。它还原一次真实执行；它不去预测这个 kernel 在另一台机器上会怎么跑。

## 2. 三类事实

还原只用三类事实，不用别的。

| 事实 | 说明 | 来源 |
|---|---|---|
| 运行时事实 | 每个 warp 做了什么：执行的指令、活跃线程、访问的地址、同步与异步操作的操作数 | 随 kernel 在真实 GPU 上运行的 tracer |
| 静态事实 | 编译器做了什么决定：每条指令的操作、读写的寄存器、承载编译器调度的控制字 | kernel 二进制 |
| 机制事实 | 硬件如何回应：延迟、发射间隔、带宽、容量、指令占用的执行单元 | 微基准，每个 GPU 型号测一次 |

运行时事实与静态事实定下执行的**是什么**。机制事实定下每件事**花多久**。trace 本身不用作时序来源：插桩会拖慢 kernel，被插桩运行的时序不是真实运行的时序。

## 3. 两条规则

- **可追溯。** 仿真出的每个周期都能回溯到一条采到的事实或一次测量。缺的事实绝不用默认值代替：运行以 `missing_fact:<fact>` 停止。模型不覆盖的情形以 `unsupported:<reason>` 停止。
- **不拟合。** 没有任何参数是为了拟合某个 workload 的周期数而选定的。没有按 kernel、库或输入形状取值的规则；机制树里的每个值都记录其来源和一个证据等级。

## 4. 两种生命周期

硬件的行为只测一次。kernel 做了什么，每次 launch 采一次。

```
once per GPU model                 for every kernel launch

microbenchmarks                    kernel on the GPU + tracer
      │                                   │  capture
      ▼                                   ▼
mechanism tree ──────────────┐     capture directory
(one per device)             │            │  convert   (one cluster at a time)
                             │            ▼
                             ├──────► segments
                             │            │  build     (one job per launch)
                             │            ▼
                             │         job.bin
                             │            │  simulate  (cycle by cycle)
                             │            ▼
                             └──────► execution record
```

| 阶段 | 定下什么 | 命令 | 设计注记 |
|---|---|---|---|
| 采集 | 发生的是哪次执行 | `cyclescope capture` | [capture.md](capture.md) |
| 转换 | 一切不依赖时间的东西：每条指令对 barrier 或搬运做了什么、触及哪些缓存扇区、将用到哪些叶 | `cyclescope convert` | [convert.md](convert.md) |
| 构建 | 把一次 launch 的各 cluster 合成一份输入 | `cyclescope build` | [job-format.md](job-format.md) |
| 仿真 | 每条指令何时发射；若发射得晚，为什么 | `cyclescope-engine simulate` | [engine.md](engine.md) |

转换一次处理一个线程块 cluster，它为一个 cluster 定下的东西不依赖另一个，所以各 cluster 并行转换。仿真不能这样拆：warp 之间相互等待，所有线程块共享 L2 缓存和 DRAM，所以一次 launch 作为一个整体推进。

## 5. 机制、轴与叶

**机制**是 kernel 能观测到、微基准能单独触发的一种硬件行为：一类指令的延迟、barrier 完成到等待者恢复之间的延迟、L2 缓存的带宽。它是从外部描述的，按 kernel 感受到的东西，而不是按背后的电路。

同一个机制在不同条件下表现不同。每个条件是一个**轴**：数据类型、形状、访问宽度、等待的 warp 数。对一次已经发生的执行来说，用到机制的这一刻每个轴值都是已知的：要么是采集到的事实或静态事实，要么是引擎维持的状态。

所以机制存成一棵树。内部节点按一个轴分叉，**叶**存放一种条件组合下的行为，形式是常量、某个轴的线性函数或一张表。每个设备有自己的树，约五十个机制、几百片叶。见 [mechanism-tree.md](mechanism-tree.md)。

## 6. 引擎

引擎把每个 warp 采到的指令流对照机制树重放，把每条指令放在硬件允许的最早周期上。

"硬件允许什么"需要一个结构。引擎假设一个小结构：SM 子分区的发射槽、执行单元、访存端口与各级缓存、张量核流水线，以及由指令集定义的 barrier 状态机。结构里不含数字。其中的每个间隔、带宽、容量和延迟都是一片叶。

能拖住一条指令的东西分三种，各自由一类事实解决。

| 什么拖住了指令 | 由什么解决 | 例子 |
|---|---|---|
| 编译器的调度 | 静态事实 | 寄存器依赖、控制字指明的 stall 数与记分板 |
| 其他 warp | 运行时事实 | barrier、借全局存储做的同步 |
| 硬件的容量 | 机制事实 | 执行单元、访存端口、各级缓存、所有 SM 共享的 L2 与 DRAM 带宽 |

引擎不计算数据值，也不仿真机制以下的东西。它按事件推进：时间分辨到周期，但从一次状态变化直接跳到下一次。Hopper 与 Blackwell 共用一个引擎；每个架构各加一个模块，处理自己的张量核与同步指令，并带自己的树。见 [engine.md](engine.md) 与 [arch-semantics.md](arch-semantics.md)。

## 7. 执行记录

仿真的输出是一份执行记录，不只是一个总数：

- 结果：总周期数、发射与完成的指令数、按原因分的停顿周期、计数器；
- 发射日志：每条已发射指令一行，含发射周期、等了多久、等待原因，同步操作还含它指向的对象；
- 等待账本：按 warp 与原因计的等待周期。

于是性能问题变成在记录里查一次，而不是从聚合计数做推断。文件布局见 [../formats.md](../formats.md)。

## 8. 东西在哪

| 部分 | 代码 | 设计注记 |
|---|---|---|
| tracer 与采集 driver | `tools/tracer/`、`python/cyclescope/capture/` | [capture.md](capture.md) |
| 转换 | `cpp/capture/`（trace → 事实）、`cpp/convert/`（事实 → 段）、`python/cyclescope/convert/`、`python/cyclescope/arch/` | [convert.md](convert.md) |
| 作业文件 | `python/cyclescope/job/`、`python/cyclescope/io/` | [job-format.md](job-format.md) |
| 机制树 | `configs/mechanism_tree/`、`configs/trees/`、`python/cyclescope/tree/`、`tools/microbench/` | [mechanism-tree.md](mechanism-tree.md) |
| 引擎 | `cpp/core/`、`cpp/arch/` | [engine.md](engine.md) |
| Hopper 与 Blackwell 差在哪 | — | [arch-semantics.md](arch-semantics.md) |
| 术语与命名规则 | — | [naming.md](naming.md) |
