# 作业文件：段与容器

[English](../../en/design/job-format.md)

**段**把一个 cluster 不依赖时间的事实和将为它们定时的机制程序打在一起。**容器**（`job.bin`）打包一次 launch：它的各段加 launch 级表。引擎读容器，别的什么都不读。

## 1. 一种文件壳

两种格式——连同结果文件——共用一个壳：定长头，后面是由表目录定位的定长记录表，目录项为 `(tag, 记录宽, 条数, 偏移)`。未知 tag 被忽略，所以加表不破坏读方。字节布局的唯一权威在 [../formats.md](../formats.md)；本页只说每张表*做什么*。

## 2. 段里有什么

`GSIMSEG`（一个 cluster），按角色分组：

| 组 | 表 | 内容 |
|---|---|---|
| 身份 | `META`、`STRS` | cluster id、计数，以及三个摘要：机制树、编译程序、转换器 |
| 驻留结构 | `CTAS`、`WARP` | cluster 的各 CTA（rank、SM id）与每个 warp 的槽位、指令区间 |
| 指令流 | `TMPL`、`INST`、`REGS`、`SEMA`、`SPLT`、`CLKM` | 按模板与按实例：执行了什么、它的寄存器、它对 barrier 与搬运做的事 |
| 存储 | `MACC`、`MSEC` | 每次访问触及的扇区 |
| 定时 | `TIME`、`USES`、`FACT` | 时间行与机制绑定，附采集到的轴值 |
| barrier 簿记 | `MBPI`、`MBOL`、`MBOB` | 预置 barrier、多目标指令的对象清单、各对象的属主 CTA |
| 观测 | `GSYN`、`CODE` | 全局存储同步值观测与代码地址 |

段能用到的机制程序随它一起走：七张共享表（`PROG`、`NODE`、`GRD `、`LEAF`、`PARM`、`PTS `、`AXIS`）在每个 segments 目录只存一份（`program.tables`，`GSIMPRG0`），读方拼接回去。段因此保持小，一次 launch 共享一份程序镜像。

## 3. 容器加了什么

`GSIMJOB`（一次 launch）加上只在 launch 范围有意义的行：

| 表 | 内容 |
|---|---|
| `LNCH` | launch 事实：grid/block/cluster 维度、共享存储大小、寄存器数、SM 数、L2 容量、占用率上限、参照 SM 时钟与缓存控制条件、设备 |
| `SEGD` | 段目录：每段的 cluster id、文件内字节区间与 sha256 |
| `PLCE` | CTA→SM 放置：每个 CTA 跑在哪个 SM、warp 槽基址 |
| `GSYN` | 各段全局存储同步观测的全 launch 合并 |
| `TOPO` | SM 拓扑：放置域、L2 分区、TPC 对、GPC |

段镜像本身附在表后，经 `SEGD` 定位。合并时把 CTA 重编号成全 launch 的稠密空间，并重设各段内索引的基址（[../formats.md](../formats.md) §4）。

## 4. 构建流程与完整性

`cyclescope build <segments-dir> --out job.bin` 收集一次 launch 的各段，检查它们的树、程序与转换器摘要完全一致（用不同树构建的段不会被悄悄合并），然后写容器。载荷与每段的 sha256 在读取时校验。

## 5. 缓存与重链

转换是开销最大的一步，所以段有缓存（`python/cyclescope/job/cache.py`）：条目按采集、cluster、转换器源码摘要与树摘要键控。命中就复用字节；转换输入一致但树不同的条目不重新转换，而是**重链**（`python/cyclescope/job/relink.py`，`cyclescope relink`）：程序表为新树重新编译，每条 `USES` 行用它存的轴值重新走一遍——采集到的事实不变。`cyclescope split` 是 `build` 的逆操作，把容器解回一个 segments 目录。

## 6. 结果

同一个壳也承载输出：`GSIMRES`（结果计数）、`GSIMJNL`（发射日志）与 `GSIMWLG`（等待账本），布局见 [../formats.md](../formats.md) §7。

## 7. 代码位置

| 部分 | 位置 |
|---|---|
| 共享壳、头、目录 | `python/cyclescope/io/{file_header,fixed_tables}.py`、`cpp/core/job_format.hpp`、`job_reader.{hpp,cpp}` |
| 段表 | `python/cyclescope/job/{segment_format,segment_builder,semantics,timing_rows,program_tables}.py`、`cpp/convert/segment_builder.{hpp,cpp}` |
| 容器写/合并/读 | `python/cyclescope/job/container.py` |
| 缓存、重链、拆分 | `python/cyclescope/job/{cache,relink}.py`、`python/cyclescope/cli.py` 的 `cyclescope split` |
| 结果写出 | `cpp/core/result_writer.{hpp,cpp}` |
