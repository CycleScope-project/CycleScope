# 转换：把所有不依赖时间的事情定下来

[English](../../en/design/convert.md)

转换把一份采集变成仿真输入，全程不问"何时"。它分两步：trace 先变成按 cluster 组织的**事实**（`cyclescope convert --capture`），事实再变成**段**（`cyclescope convert --facts`）。工作单位是一个线程块 cluster，为一个 cluster 定下的东西不依赖其他任何 cluster。

## 1. 只用事实的规则

一条动态指令的事实只来自它自己的记录，或以它的 id、操作数值为键的记录。跨 warp 不做任何推断：一次等待观测到的是哪一代 barrier、一次读看到的值是哪次写产生的，这里都不判定。barrier 代际与跨线程块配对由引擎在仿真时求解——generation 字段离开转换时标记为未解。缺的事实不用默认值补：转换以错误停止（§8）。

## 2. 第一步：trace → 事实

记录流被回放进一个类型化模型（`python/cyclescope/convert/model.py`；C++ 侧在 `cpp/capture/`）：

- **launch 事实**：grid/block/cluster 维度、设备、共享存储大小、寄存器数、CTA→SM 放置、参照时钟；
- **静态指令**：解码出的 SASS——opcode、operation class、源与目的寄存器、控制字；
- **动态指令**：每次执行一条，带活跃掩码、谓词结果与操作数事实；
- **访存引用**：地址空间、访问类别、宽度与逐 lane 地址；
- **warp 流**：每个 warp 的动态指令按程序序排列；
- **cluster 单元**：一个 cluster 的各 CTA，连同其 warp 流、tensor map 与预置 barrier。

轮询折叠与全局存储同步观测也在这里落地：失败轮询按 [capture.md](capture.md) 所述折叠；每次强全局访问处观测到的值作为观测记录带过去，留给引擎配对。输出是每个 cluster 一个 `GSIMCAP` 事实文件，外加一个 `launch.json` 边车文件。

## 3. 第二步：事实 → 段

每个 cluster，构建器写出一个段。每条动态指令得到的东西（布局见 [../formats.md](../formats.md)）：

| 表 | 含义 |
|---|---|
| `TMPL`/`REGS` | 指令模板：operation class、源与目的寄存器、记分板与控制字 |
| `INST` | 实例本身：用哪个模板、哪条语义行、哪次访存 |
| `SEMA` | 这条指令对 barrier、组或搬运*做了什么*：对象、计数、tensor map、描述符——解开的操作数事实 |
| `MACC`/`MSEC` | 它的访存触及哪些缓存扇区（按访问合并的扇区清单） |
| `TIME` | 时间行：在哪个程序事件上用哪片叶 |
| `USES`/`FACT` | 绑定：适用哪个机制程序，沿守卫树走到剩余轴都归引擎的那个节点，附采集到的轴值 |

其余由辅助表携带：`CTAS`/`WARP`（cluster 成员与 warp 槽位）、`MBPI`/`MBOL`/`MBOB`（预置 barrier 与多对象指令）、`GSYN`（全局存储同步观测）、`CODE`（取指模型用的代码地址）、`STRS`（字符串）。

## 4. 提前选好叶

每个机制程序由若干轴守卫。转换解析所有 owner 为 `capture`、`converter` 或 `launch` 的轴——数据类型、访问宽度、operation class、cluster 形状——用的轴函数在 `python/cyclescope/convert/axes.py`（`cpp/convert/axes.cpp`）。走到一个归 `engine` 所有的轴时——比如已到达某 barrier 的 warp 数这类只在仿真中存在的状态——转换停下，写一条钉在该节点上的 `USES` 行；引擎从那里继续走。某个守卫没有匹配的孩子，说明该程序不适用；capture 轴缺值则是 `missing_fact`。轴的 owner 按轴声明在 `configs/mechanism_tree/*/axes.json`。

## 5. 架构差异从哪一侧进入

转换共用一个模型；架构专有的部分是可插拔的规则：操作数事实的解释、有效 operation class、语义解码在 `python/cyclescope/arch/<arch>/` 与 `cpp/arch/<arch>/{capture_rules,convert_rules}.cpp`。见 [arch-semantics.md](arch-semantics.md)。

## 6. 两个构建器

C++ 构建器是默认（`--builder cpp`；第一步用 `cyclescope-capture-facts`，第二步用 `cyclescope-convert`）。Python 构建器（`--builder python`）是参照实现。`tests/common/test_convert_cpp_builder.py` 用两个构建器转换同一份事实，断言两者必需段表逐字节相同，于是一处语义改动无法只落在一个构建器里而不被测试发现。

## 7. 并行与分片

cluster 之间互不依赖，所以转换按 cluster 分片，分片并行跑；`cyclescope-trace-split` 可预先切分大的记录流。命令与输出布局见 [../pipeline.md](../pipeline.md)。

## 8. 失败

数据问题只有两类失败：

- `missing_fact:<fact> (<where>)`——模型需要的事实不在采集里，例如 barrier 等待的相位操作数没采到时是 `missing_fact:parity`，命名 barrier 的计数存在没被记录的寄存器里时是 `missing_fact:cta_barrier_operand`；
- `unsupported:<reason> (<where>)`——事实存在但模型不覆盖，例如一个 warp 把状态机不跟踪的几种 async 组混在一起时是 `unsupported:mixed_async_groups`。

应用与资产错误（文件损坏、摘要不符）是格式错误，不进入这套词汇。

## 9. 代码位置

| 部分 | 位置 |
|---|---|
| trace → 事实（C++） | `cpp/capture/`（`trace_reader`、`control_flow`、`memory_refs`、`operand_facts`、`poll_folding`、`global_sync`、`tensor_map`、`static_code`、`facts_writer`、`launch_doc`） |
| 事实 → 段（C++） | `cpp/convert/`（`facts_reader`、`segment_builder`、`semantics`、`traversal`、`axes`、`trace_split`） |
| 同样的两步，Python 版 | `python/cyclescope/convert/`（`model`、`trace_reader`、`static_code`、`control_flow`、`poll_folding`、`operand_facts`、`global_sync_observation`、`tensor_map`、`semantics_common`、`mechanism_uses`、`axes`、`facts_format`、`pipeline`、`capture_pipeline`） |
| 架构规则 | `python/cyclescope/arch/<arch>/`、`cpp/arch/<arch>/` |
