# 微基准：测量机制事实

[English](../en/microbench.md)

机制叶是单独测量的：一个小 kernel，让被测机制成为唯一变化的开销——扫轴或重复次数，读斜率，固定开销就此消去。cluster-local 机制在只有一个 SM 活跃时测，gpu-shared 在整卡有负载时测。什么是机制与叶：[design/mechanism-tree.md](design/mechanism-tree.md)。

## 1. 一个协议目录

每个探针住在 `tools/microbench/<name>/`（不是每个目录都有全部文件）：

| 文件 | 作用 |
|---|---|
| `protocol*.json` | 协议，测量之前写好：测什么、测点、判据 |
| `probe/` | 被测的 CUDA kernel |
| `build.sh` / `run.sh` | 编译与运行探针（另有 `run_probes.sh`、`run_ncu.sh` 变体） |
| `model.py` | 把原始计时变成协议统计量 |
| `sass_check.py` | 检查编译出的 SASS 确实是探针意图 |
| `install.py` / `table*.py` | 把结果落到树的片段上 |

## 2. 一次测量的生命

1. 先写协议——测点、控制与判据在探针运行前冻结。
2. 编译并检查 SASS 确实是协议要的测量。
3. 在每款设备上跑，各两次。
4. 用协议判据判定结果。
5. 以相应证据等级装值，或者保持叶不变并记录原因。

## 3. 本地跑一个

```bash
env CYCLESCOPE_ARTIFACT_ROOT=<artifact root> \
    PROTOCOL=<name> DEVICE=h100|b300 RUN=<tag> \
    [CELL=<subdir under tools/microbench>] [OUT_ROOT=<dir>] \
    [NCU=<ncu>] [TIMEOUT=3600] \
    bash tools/microbench/run_local.sh
```

脚本在 `tools/microbench/<dir>` 内跑 `run.sh`（否则 `run_probes.sh`，否则 `run_ncu.sh` 并传 `NCU`）——`<dir>` 给了 `CELL` 就是 `CELL`，否则是 `PROTOCOL`。输出到 `$CYCLESCOPE_ARTIFACT_ROOT/microbench/<dir>/<device>-<run>`（或 `OUT_ROOT`）。一切在 `timeout` 下运行；`TREE` 与 `SM_ARCH` 为探针设好，`WGMMA_*`、`UMMA_*`、`V2`、`V3`、`SASS_CHECK`、`RUNS`、`CELLS`、`SKIP_BUILD`、`KINDS`、`PROBE`、`SET` 透传。成功打印 `CELLS_DONE`；失败的测点打印 `CELL_FAIL`。

## 4. 目录

`tools/microbench/` 下每个目录一行，按字母序。最后一列列出在任一设备的树里、有值引用了该目录来源的机制。"—" 表示没有已装入的值引用它：这次测量印证了已装入的值、成了引擎里的一条规则，或者没有装入。

| 目录 | 测什么 | 有值来自它的机制 |
|---|---|---|
| `alu_forms` | 整数 ALU 各形式的逐指令开销 | `instruction.service` |
| `arrive_status` | `SYNCS.ARRIVE` 返回的状态何时可用 | `instruction.service` |
| `atomic_issue` | 原子指令的每 SM 发射串行 | `storage.atomic.issue` |
| `barrier_release` | mbarrier / CTA / cluster / cp.async barrier 的释放延迟 | `instruction.service`, `synchronization.cluster.release`, `synchronization.cp_async.arrival`, `synchronization.cta.release`, `synchronization.mbarrier.release` |
| `constant_launch` | 首次遍历的常量缓存 miss 开销 | `memory.constant_cache` |
| `control_branch` | 控制流分支开销 | `control.branch` |
| `control_word` | 固定延迟指令类的延迟，由编译器给出的控制字推导 | — |
| `control_yield` | `YIELD` 给 warp 的发射下限 | `instruction.service` |
| `crossings` | 一组小探针：常量缓存、CTA barrier 释放、描述符重取、L1 带宽与驻留、重汇合、store 端口、try-wait 超时 | `control.reconvergence`, `memory.constant_cache`, `memory.dram.service`, `memory.l1.residency`, `memory.l1.service`, `memory.l2.service`, `memory.store_dram.service`, `memory.store_l2.service`, `storage.tma.descriptor_refetch`, `synchronization.mbarrier.trywait_timeout` |
| `descriptor_visibility` | 异步描述符 publish → acquire 可见延迟 | `synchronization.descriptor.visibility` |
| `dispatch` | 逐 CTA 的 launch 分派串行 | `launch.dispatch` |
| `dispatch_contention` | 分派下的发射争用（寄存器堆操作数读、fma 发射） | `instruction.service`, `register_file.operand_read` |
| `dispatch_refill` | 释放出的 CTA 槽位空多久 | `launch.dispatch` |
| `dsmem` | cluster 内远程共享存储访问开销 | `storage.shared.acceptance_sm`, `storage.shared.issue`, `storage.shared.visibility` |
| `ffma2_operands` | Blackwell 上 FFMA2/FADD2/FMUL2 吞吐 | — |
| `fp32_mix` | 一个 SMSP 上 FP32 流与 MUFU 交叠 | — |
| `fp_stream_mix` | 每 SMSP 若干 warp 的 FP32/MUFU 混合 | — |
| `gather_hit` | 对同一行的背靠背 gather 读的等待 | `memory.load_port` |
| `gather_latency` | 多 warp 多 SM gather 同一行时的读延迟 | — |
| `global_atomic` | 全局原子的 release-to-acquire 延迟 | `memory.global_atomic` |
| `ifetch` | L0/L1 取指 miss 与预取行为 | `instruction.fetch` |
| `l2_merge` | 同一 L2 miss 上多个读的合并 | `memory.l2_merge` |
| `ldgsts_l1` | 哪些 cp.async 填充在 SM 的 L1 里分配 | — |
| `ldgsts_loaded` | cp.async 负载下的 L1/L2 服务 | `memory.l2_merge` |
| `lifecycle` | launch 包络：进入、退出、排空 | `kernel.lifecycle` |
| `load_after_copies` | 非 bypass 异步拷贝如何把 local 行挤出 L1 | `memory.l1.service` |
| `load_after_stores` | 等 store 的栈重读开销 | `memory.store_l2.service` |
| `load_lines` | lane 触多条 L1 行时的读开销 | — |
| `load_port_lines` | SM 读端口按扇区还是按行计费 | — |
| `lsu_accept` | 存储层级前的接受队列与 store 发射 | `storage.global.store_issue` |
| `math_occupancy` | warp 交叠两类指令时的逐指令开销 | `instruction.service` |
| `memlat` | 各缓存级的 load-to-use 延迟 | `memory.l1.service`, `memory.l2.service` |
| `memlat_loaded` | 全卡负载下的 L2/DRAM 延迟 | `memory.dram.service`, `memory.l2.service` |
| `memory_request` | 各级每 SM 饱和请求率，含 store | `memory.dram.capacity`, `memory.l1.capacity`, `memory.l2.capacity` |
| `mma_sync` | warp 级 mma.sync 类的计时 | `instruction.service` |
| `mufu_rcp` | `MUFU` 倒数的吞吐 | — |
| `nanosleep` | NANOSLEEP 时长与 `.SYNCS` 唤醒行为 | `control.nanosleep` |
| `register_operands` | 寄存器源形式怎么改变发射间隔 | — |
| `row_gather` | 读延迟随每 warp DRAM 行数的变化 | `memory.dram.service`, `memory.dram_partition` |
| `service_latency` | 各类的 instruction.service 延迟字段 | `instruction.service` |
| `shared` | 共享存储读写开销、wavefront、fence | `instruction.service`, `storage.shared.acceptance_sm`, `storage.shared.acceptance_smsp`, `storage.shared.fence`, `storage.shared.issue`, `storage.shared.visibility` |
| `shfl_scope` | SHFL 发射作用域（SMSP 还是 SM） | `instruction.service` |
| `store_flood` | 饱和 store 的每 SM 与全卡开销 | `memory.store_l2.capacity`, `memory.store_l2.service` |
| `store_hold` | 后续指令何时能覆盖 store 的数据 | — |
| `store_scatter` | store 发射间隔取决于宽度还是生成的行 | `storage.global.store_issue` |
| `tensor_pipe` | 每 SM 张量管线节拍与完成 | `tensor.completion`, `tensor.pipe` |
| `tensor_pipe_forms` | 各 MMA 形式下的同一节拍 | `tensor.completion`, `tensor.pipe` |
| `tma_cold` | TMA 完成的冷热差异 | `storage.tma.completion` |
| `tma_forms` | 各指令形式下 TMA 的服务/完成 | `storage.async.service_start`, `storage.ldgsts.completion`, `storage.tma.completion` |
| `tma_loaded` | 全卡 TMA 负载下的 L2→SM 投递 | — |
| `tma_multicast` | multicast tile 读向目的 CTA 的投递 | — |
| `tma_oob` | 部分越界 TMA 读的额外延迟 | `storage.async.service_start`, `storage.tma.completion` |
| `tma_store_loaded` | TMA store 计时与共享 store/L2 端口争用 | `memory.shared_service`, `memory.store_l2.service`, `storage.tma.reduce` |
| `tmem` | 张量存储加载（`LDTM`）的发射间隔 | `blackwell.tmem.load`, `instruction.service` |
| `tmem_pipeline` | TMEM 拷贝/读的管线行为 | `blackwell.tmem.copy`, `tensor.pipe` |
| `trywait_wake` | 失败的 try_wait 之后的唤醒延迟 | `synchronization.mbarrier.release` |
| `wgmma_handoff` | 从 wgmma 到消费者的累加器交接 | `hopper.wgmma.accumulator_handoff.consumer` |
| `wgmma_overlap` | wgmma 在途时的发射开销 | `hopper.wgmma.register_file_issue_steal` |

## 5. 写一个新探针

- 先写协议，后测量。
- 一定检查编译出的 SASS——编译器会把放错位置的计时窗口重写成协议没打算测的东西。
- 隔离一个机制：让它成为各测点间唯一变化的开销，并把控制项（热路径、空操作变体）放进同一 kernel。
- 扫重复次数或轴、读斜率；单个点会把固定开销混进测量值。
- 按归属选测量形态：cluster-local 一个 SM 活跃，gpu-shared 整卡有负载。
