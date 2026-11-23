# 读结果

[English](../en/results.md)

一次运行产出两类输出：报告行——告诉你仿真周期数离硬件多近；以及执行记录——告诉你*为什么*——每条指令的发射周期与每个 warp 在等什么。

## 1. 报告

```bash
cyclescope report <run_out> [<run_out> ...]
    [--expected expected.tsv] [--tolerance PCT] [--out file]
cyclescope report --baseline <expected.tsv>
```

`report` 读取每个运行输出目录下所有的 `sim/<case>/status.json`，每个用例打印一行，制表符分隔：

| 列 | 含义 |
|---|---|
| `suite`、`case` | 哪个用例 |
| `stage` | 运行停在哪一阶段，或 `done` |
| `class` | `ok` 或失败类别 |
| `issued` | 引擎发射的指令实例数 |
| `sim_cycles` | 仿真周期数 |
| `hw_cycles` | 同一 launch 在 GPU 上的周期数（Nsight Compute） |
| `ratio` | `sim_cycles / hw_cycles` |
| `err_pct` | `abs(ratio - 1)`，百分数 |
| `base_sim` | 随发基线的 `sim_cycles` |
| `delta_pct` | 本次运行相对基线的距离，带符号百分数 |
| `placement` | 采集的 CTA→SM 放置摘要 |
| `verdict` | 一个词——见下 |

verdict 把这次运行与随发的 `expected.tsv` 比较：

- `ok`——运行完成，且在同一放置下与基线相差不超过 `--tolerance` 个百分点（默认 3 %）。
- `fail`——运行没有干净完成，或超出容差。任一行 `fail` 时 `report` 退出码为 1。
- `listed`——运行完成但放置与基线不同，基线数字不直接适用。
- `n/a`——该用例不在期望表里。

`--expected` 指向别的期望表（可重复）；不给就用套件自己的 `expected.tsv`。`--out` 把表写进文件。各行之后，每个套件一段 `#` 块打印精度行（`cases`、`mape`、`p90_abs_err`、`median_ratio`——在带硬件周期数的完成行上计算）和基线行（`within`、`beyond`、`listed`、最大 |delta|）。

`--baseline <expected.tsv>` 单独打印一张随发期望表的精度摘要——不需要运行。

## 2. 执行记录

引擎最多写三个文件：

| 文件 | magic | 内容 |
|---|---|---|
| `result.bin` | `GSIMRES` | 总量：周期、issued/completed、硬失败、逐 SMSP 活跃区间、按桶的等待周期、资源等待、遥测计数器 |
| `journal.bin` | `GSIMJNL` | 每条发射实例一行：warp、SMSP、实例、发射周期、operation class、模板、等待桶、等待周期、同步对象 |
| `wait.bin` | `GSIMWLG` | 等待账本：按 warp 与原因的总周期数与次数 |

`cyclescope run` 把 `result.bin` 写到 `sim/<case>/out-<builder>/`；`--ledgers` 在同目录加上 `journal.bin` 与 `wait.bin`。手工运行则来自 `cyclescope-engine simulate --journal <path> --wait-ledger <path>`；`--stall-ledger <path>` 再写出逐 PC 的 stall 账本（`stall.tsv`，schema `cyclescope.stall-ledger.v1`；engine.md §4）。

全部 1,000 个套件 kernel 的发射日志与等待账本发布在 [CycleScope-records](https://github.com/CycleScope-project/CycleScope-records)，不用运行就能研究。

读取器在 `cyclescope.job.result`——`read_result`、`read_journal`、`read_wait_ledger`，各收文件字节：

```python
from pathlib import Path
from cyclescope.job.result import (
    read_result, read_journal, read_wait_ledger)

res = read_result(Path("result.bin").read_bytes())
print("cycles:", res["cycles"], " issued:", res["issued"])

waits = {}
for row in read_journal(Path("journal.bin").read_bytes())["JRNL"]:
    if row["wait_bucket"] != 0xFFFFFFFF:          # 0xFFFFFFFF = no wait
        waits[row["wait_bucket"]] = \
            waits.get(row["wait_bucket"], 0) + row["wait_cycles"]
for bucket, cycles in sorted(waits.items()):
    print(f"bucket {bucket}: {cycles} cycles")
```

## 3. 等待桶

日志的 `wait_bucket` 与账本的 `cause_kind` 存 `cpp/core/simulator.hpp` 的 `WaitBucket` 枚举——`0xFFFFFFFF` 表示这条指令没有等待：

| id | 桶 | warp 在等什么 |
|---|---|---|
| 0 | dispatch | CTA 被分派、warp 进入 |
| 1 | fetch | 取指 |
| 2 | collective | barrier 释放或挂起唤醒 |
| 3 | semantic | 语义门（一次轮询、一个异步组、一条 arch 规则） |
| 4 | resource | 忙碌的执行单元、端口、队列或发射间隔 |
| 5 | register | 寄存器值或编译器排程 |
| 6 | scheduler | 不归因于以上检查：warp 首次检查失败前的初始原因 |

## 4. 接下来看哪

- [formats.md](formats.md) §7——三个文件的字节布局。
- [engine.md](engine.md)——各遥测组计的是什么。
- [design/engine.md](design/engine.md)——产生它们的模型。
