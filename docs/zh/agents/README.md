# 代理调试手册

[English](../../en/agents/README.md)

供代理（自动化调试工具）排查采集、转换与仿真问题的操作手册。
先读本页，再按症状进入对应手册。

## 基本原则（来自项目原则）

1. **机制，不查表**：不按 case id、kernel 名、shape 或库写时间规则或拟合叶值。只对一个 case 有效的改动不是修复。
2. **采集只记事实，转换只搬事实**：动态指令的事实只来自它自己的记录或以它的 id / 操作数为键的记录。mbarrier 与 CTA barrier 的代际由引擎在仿真时求解，绝不在转换时推断。
3. **缺事实就拒收，不补默认值**：数据失败只有两类 —— `missing_fact:<fact> (<where>)` 与 `unsupported:<reason> (<where>)`。不要为让 case 通过而发明回退值。
4. **每叶带证据等级与来源**（`measured / experimental / structural / placeholder`，见 `configs/mechanism_tree/<arch>/sources.json`）；补测先写冻结协议（`tools/microbench/<name>/protocol.json`），判据不过就不装。
5. **先归因后装叶**：不要因为某 case 误差变小而选机制或叶值。
6. **纯重构保持结果逐字节不变**：用 `cyclescope run` 跑出货示例与 `suites/<arch>/full_gpu/expected.tsv` 验证。

## 阶段图与各阶段输出位置

`cyclescope run --suite <s> --case <c> --out <out>` 产出：

```
<out>/captures/<case>/         采集阶段（run.json、trace-*.bin.zst、
                               launches.jsonl、functions/、app.log、
                               ncu_launches.jsonl、trace_check.txt）
<out>/captures/reference/<case>/  参照阶段（reference.tsv、timing.json、
                               ncu 重复）
<out>/sim/<case>/              convert.log、segments.log、build.log、sim.log、
                               stages.tsv、status.json、facts/launch.json、
                               out-<builder>/（job.bin、result.bin、journal、
                               wait ledger）
```

终态 `stage` / `class` 在 `sim/<case>/status.json`；`stages.tsv`
每阶段一行 `stage wall_s max_rss_kb rc cpu_s`。`cyclescope run`
只有 `stage=done, class=ok` 才退出 0。

## 分诊表

| 症状（阶段 / class 或出现位置） | 手册 |
|---|---|
| 要采集不在套件里的 kernel | [new-kernel.md](new-kernel.md) |
| 要采集从未见过的库/框架 | [new-library.md](new-library.md) |
| `capture`/`reference` 阶段；`app_failed:`、`ncu_failed:rc=…`、`ncu_timeout`、`selection_failed:`、`zstd_missing`、`bad trace header`、`no traced launch`、`missing trace-N.bin.zst`、`missing functions/<id>.tsv` | [capture-failures.md](capture-failures.md) |
| `convert`/`segments` 阶段；`launch.json`/日志中的 `missing_fact:*`、`unsupported:*` | [convert-failures.md](convert-failures.md) |
| `sim` 阶段；`unsupported:engine:*`、`deadlock`、`crash`、`issued_ne_completed`、硬约束非零 | [engine-failures.md](engine-failures.md) |
| 机制存在但叶缺失/占位/`unsupported:engine:discrete_miss` | [missing-leaf.md](missing-leaf.md) |
| 硬件行为完全没有机制规则 | [missing-mechanism.md](missing-mechanism.md) |

## 手册索引

- [new-kernel.md](new-kernel.md) —— 采集 + 仿真不在套件中的
  kernel
- [new-library.md](new-library.md) —— 拉起从未采集的库/框架
- [capture-failures.md](capture-failures.md) —— tracer / census /
  参照失败
- [convert-failures.md](convert-failures.md) —— 转换期
  `missing_fact:` / `unsupported:`
- [engine-failures.md](engine-failures.md) —— 模拟器错误与死锁
  分诊
- [missing-leaf.md](missing-leaf.md) —— 缺失/占位叶参数
- [missing-mechanism.md](missing-mechanism.md) —— 端到端加机制
