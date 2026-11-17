# 机制树

[English](../../en/design/mechanism-tree.md)

**机制**是 kernel 能观测到、微基准能单独触发的一种硬件行为：一类指令的延迟、barrier 完成到等待者恢复之间的延迟、L2 缓存的带宽。每个让行为发生改变的条件是一个**轴**；一片**叶**存放一种条件组合下的行为。每个设备有一棵生成的树，装下全部机制。

## 1. 一个机制和它的叶

内部节点按一个轴分叉；守卫按一个值或一个区间选孩子。在 B300 上，`tensor.pipe`——一次张量操作占用共享张量流水线的间隔——依次按 `operation_class`、`umma_kind`、`cta_group`、`a_src` 分叉：

```
tensor.pipe
└── operation_class = UTCHMMA
    └── umma_kind = f16
        └── cta_group = 1            (one CTA, not a pair)
            └── a_src = smem  ──► leaf
```

该叶的 `cycles` 字段是一张按操作形状轴（`tensor_m`、`tensor_n`、`tensor_k`、`umma_operation`…）索引的离散表：

| M | N | K | operation | cycles |
|---|---|---|---|---|
| 32 | 64 | 16 | ws | 28 |
| 32 | 128 | 16 | ws | 44 |
| 64 | 8 | 16 | normal | 23 |

机制被用到的时刻，每个轴值都是已知的——要么是采集到的事实或静态事实，要么是引擎维持的状态——所以走树总落在一片叶上。

## 2. 叶里有什么

叶是 `configs/schema_registry.json` 中某个参数模式的一组命名**字段**：`cycles`、`enabled`、延迟、带宽、容量、资源 id。每个字段的值是一个模型：

| 模型 | 含义 |
|---|---|
| `constant` | 一个数 |
| `linear` | 在一个轴上穿过实测点的分段线性曲线，端点之外取端点值 |
| `affine` | 声明定义域内的 `slope · axis + intercept` |
| `discrete` | 按一个或多个轴索引的表；查不到就按失败关闭 |

叶还记录一个**作用域**（值描述的实体：一条指令、warp、CTA、SMSP、SM、cluster 或整卡）和单位。以纳秒声明的字段，在引擎解析时按 launch 的参照 SM 时钟（`reference_sm_clock_khz`）换算成周期（`cpp/core/mechanism_runtime.cpp`）。

## 3. 值从哪来

每个字段点名它的**来源**——`configs/mechanism_tree/<arch>/sources.json` 里的键，每个来源记录一个证据等级和一段测量说明：

| 证据 | 含义 |
|---|---|
| `measured` | 按冻结协议的规则从微基准读出 |
| `experimental` | 实测过，但协议判据只部分满足（例如两次运行分歧超过界限）——原因写在来源里 |
| `structural` | 由构造决定：模型自身的常量（使能位、单位换算、假设的结构） |
| `placeholder` | 协议测到之前的占位值 |

叶的状态由来源推导：全为 structural → `structural`；measured 与 structural 混合 → `measured`；含 experimental → `experimental`；否则 `placeholder`。设备条目也可以是 `unsupported`：该类在该架构上不存在（例如 B300 树中 Hopper 的 `HGMMA`）；保留这些节点是为了让划分完整。

测量的通常形态：一个小 kernel，被研究的机制是其中唯一变化的开销；扫描重复次数或轴本身，从斜率读出数值，固定开销因而消掉；协议先写下并冻结（`tools/microbench/<name>/protocol*.json`），数值再入树。协议目录见 [../microbench.md](../microbench.md)。

## 4. 归属：cluster-local 与 gpu-shared

一片叶能否被*观测到*取决于卡上还有什么。整卡上只有一个 cluster 时，没有 L2 或 DRAM 争用、没有原子争用、没有第一波以后的分派——值依赖其他 SM 负载的叶是 **gpu-shared**，要在整卡有负载时测量。在一个 SM 或 cluster 内部的叶是 **cluster-local**。

| 归属 | 机制 |
|---|---|
| cluster-local | `instruction.*`、`scheduler.policy`、`register_file.*`、`control.*`、`synchronization.*`、`tensor.*`、`hopper.*`、`blackwell.umma.*`、单线程块预算内的 `blackwell.tmem.*`、`storage.shared.*`、`storage.async.*`、`storage.tma.*`、`storage.atomic.issue`、`memory.l1.*`、`memory.constant_cache`、`memory.layout`、只在一两个 SM 上运行的 launch 的 `kernel.lifecycle`，以及 `memory.l2.service`、`memory.dram.service`、`memory.store_l2.service`、`memory.store_dram.service` 中的每 SM 端口速率与无争用延迟 |
| gpu-shared | `memory.l2.capacity`、`memory.l2.residency`、`memory.dram.capacity`、`memory.store_l2.capacity`、`memory.store_dram.capacity`、`memory.shared_service`、`memory.l2_merge`、`memory.l2_partition`、`memory.global_atomic`、`launch.dispatch`、`memory.l2.service` 与 `memory.dram.service` 中随负载变化的延迟、跨多个 SM 的 `kernel.lifecycle`、多个 SM 竞争时的 `blackwell.clc.*`、跨线程块共享的 `blackwell.tmem.*` 预算 |

划分可以穿过同一个机制：`memory.l2.service` 把每 SM 端口速率（cluster-local）和随负载变化的延迟（gpu-shared）放在一起。未列出的机制按同一判据归类。

## 5. 什么时候取叶

每个机制声明一个程序**事件**——引擎向它要值的时刻——和一棵守卫树。事件有 `initialization`（加载时一次）、`issue` 与 `memory_issue`（指令发射时）、`barrier_release`（barrier 相位完成时）、`control_transition`（分支或重汇合）与 `collective_launch`（warpgroup 级操作上张量流水线）。转换时，走树解析所有 owner 为 `capture`、`converter` 或 `launch` 的轴，在剩余轴都归 `engine` 的地方停下；运行时引擎用它维持的状态把剩下的走完。见 [convert.md](convert.md) §4。

## 6. 源码布局

```
configs/mechanism_tree/
  common/        mechanisms, axes, resources, sources, class_resource:
                 shared definitions
  hopper/        per-arch axes + leaf values for Hopper
  blackwell/     the same for Blackwell, plus its own mechanisms
configs/schema_registry.json     parameter schemas (fields, units)
configs/trees/{h100,b300}.json   the generated trees
```

`cyclescope tree build` 把 common 与 arch 片段合并成 `configs/trees/<device>.json`；`--emit-cpp` 另外重新生成 `cpp/core/generated/parameter_schema.hpp` 与 `python/cyclescope/tree/generated_schema.py`，引擎与构建器用它们按 id 取字段。`cyclescope tree compile` 产出二进制程序表（`GSIMPRG0`），转换器拿它就不必走 JSON。

## 7. 校验与 strict 门

`cyclescope tree validate --strict` 构建两棵树并检查结构有效性（守卫把轴的定义域划分成不重不漏的分区、模型声明的轴存在且类型正确）和五类约定（`python/cyclescope/tree/conventions.py`）：

- `field_model`——共享机制的每个设备叶字段使用该机制在 `field_models` 中声明的模型类型与轴；
- `discrete_axis`——`discrete` 表只以枚举轴为键（指令编码固定的量）；连续量用 `affine` 或 `linear`；
- `unit`——纳秒字段只出现在 `memory.shared_service`、`memory.global_atomic` 与 `launch.dispatch`；
- `class_resource`——`instruction.service` 叶的资源与 `common/class_resource.json` 中的类绑定一致；
- `evidence_crossing`——共享机制不得在一台设备上有实测而在另一台是占位。

五个清单全空才算过。其他树命令：`tree show <tree> <mechanism>` 打印一棵守卫树；`tree diff` 比较两棵生成的树。

## 8. 改叶或加叶

1. 测量：在冻结协议下跑微基准（[../microbench.md](../microbench.md)）。
2. 在 `configs/mechanism_tree/<arch>/sources.json` 加带来源等级的来源条目。
3. 在 arch 的 `leaves/` 或 `mechanisms/` 片段里写叶值与其来源键。
4. `cyclescope tree build`，然后 `cyclescope tree validate --strict`。
5. 若改了模式登记表（新字段或单位），用 `tree build --emit-cpp` 重建，让生成的模式保持同步。

## 9. 代码与测试

| 部分 | 位置 |
|---|---|
| 构建、合并、摘要 | `python/cyclescope/tree/{build,source}.py` |
| 校验 | `python/cyclescope/tree/{validate,conventions,registry}.py` |
| 生成的模式 | `cpp/core/generated/parameter_schema.hpp`、`python/cyclescope/tree/generated_schema.py` |
| 运行时选叶 | `cpp/core/mechanism_runtime.{hpp,cpp}` |
| 转换侧走树 | `python/cyclescope/convert/mechanism_uses.py`、`cpp/convert/traversal.cpp` |
| 测试 | `tests/common/test_generated_schema.py`、`test_tree_validate*`（`tests/`）、`cpp/tests/mechanism_runtime_test.cpp` |
