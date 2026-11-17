# 术语与命名规则

[English](../../en/design/naming.md)

一个概念一个词，处处一致：代码、格式、配置、文档。本页是一份对照表，把各设计注记里用的概念词对到代码和二进制格式里实际的名字，外加代码遵循的命名规则。

## 1. 术语表

| 概念 | 代码 / 格式中的名字 |
|---|---|
| 线程块 | `CTA`（CTA id、`cta_index`、`%cluster*` rank） |
| 一起采集、一起转换的线程块组 | `cluster` → 一个 `GSIMSEG` 段，`cluster_id` |
| 属于一次 kernel launch 的一切 | `launch`、`LNCH` 行、容器 `GSIMJOB`（`job.bin`） |
| tracer 的每条指令记录 | `trace-<launch>.bin.zst` 中的 `record` |
| 按 cluster 转换出的事实 | `GSIMCAP` 事实文件（`*.gsc`）+ `launch.json` |
| 编译器写下的静态指令 | `StaticInstruction`；作业中是 `TMPL` + `REGS` |
| 它的一次执行 | `InstructionInstance`、`INST` |
| 实例对 barrier/组/搬运做的事 | `InstanceSemantics`、`SEMA` kind + 字段 |
| 一次访存触及的扇区 | `MACC` / `MSEC` 行 |
| 一个实例的机制绑定 | `USES` 行 + `FACT` 轴值 |
| 某个事件上的取叶 | `TIME` 行（kind）/ `LEAF` + `PARM` |
| 机制 / 轴 / 叶 | `PROG` 程序、`AXIS` + `GRD ` 守卫、`LEAF` |
| CTA→SM 指派 | `placement`、`PLCE` 行 |
| SM 互连布局 | `sm topology`、`TOPO` 行 |
| 借全局存储做的跨 CTA 同步 | `global sync`、`GSYN` 行 |
| 引擎输出 | `result` `GSIMRES`、`issue journal` `GSIMJNL`、`wait ledger` `GSIMWLG` |
| 失败类 | `missing_fact:<fact> (<where>)`、`unsupported:<reason> (<where>)` |
| 叶的证据 | `measured` / `experimental` / `structural` / `placeholder` |
| 叶的归属 | `cluster-local` / `gpu-shared`（[mechanism-tree.md](mechanism-tree.md) §4） |
| 参照条件 | `reference_sm_clock_khz`、`reference_cache_control` |

## 2. 代码遵循的命名规则

- 标识符用英文；文档 `lower-kebab-case.md`；Python 与 C++ 文件 `snake_case`；JSON 配置 `lower_snake_case.json`。
- Python：模块与函数 `snake_case`，类 `CapWords`，常量 `UPPER_SNAKE`；包名 `cyclescope`。
- C++：类型 `CapWords`，函数与变量 `snake_case`，常量 `kCapWords`；命名空间 `cyclescope`（生成的参数字段在 `cyclescope::core`）与 `cyclescope_convert`（采集/转换侧）。
- 表 tag 是四个大写 ASCII 字符（`TMPL`、`SEMA`、`GRD `）；文件 magic 是 `GSIM` + 类别 + 版本（`GSIMSEG`、`GSIMJOB`、`GSIMRES`、`GSIMCAP01`、`GSIMPRG0`）。
- CLI：一个入口 `cyclescope <verb>`；二进制 `cyclescope-{convert,capture-facts,trace-split,engine}`；环境变量 `CYCLESCOPE_*`。
- 字段名带单位：`*_cycles`、`*_ns`、`*_bytes`、`*_per_cycle`、`*_per_ns`、`*_count`、`*_index`；布尔用肯定形式（`enabled`、`is_*`、`has_*`）。
- 名字说明一个东西是什么，不说明它从哪来：不带日期、轮次号、人名、case 名，也不自造缩写。
