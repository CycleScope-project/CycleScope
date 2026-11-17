# 采集：运行时事实与静态事实

[English](../../en/design/capture.md)

tracer 只有一个任务：把已经发生的那次执行确定下来，使它的任何部分都不必靠建模得到。它记录的是"哪些指令被执行了、每条指令在等什么"所需的东西；所有时长都留给其他两类事实。

## 1. 采集什么

每条被执行的指令、每个 warp 各产生一条记录。记录带上静态指令的身份、活跃线程掩码，以及这条指令后面要用到的部分：

| 事实 | 记录内容 |
|---|---|
| 指令流 | 每个 warp 按执行顺序：执行了哪条静态指令、哪些 lane 活跃 |
| 访存地址 | 每次 load、store、atomic、async copy 与 TMA 操作数的逐 lane 地址；地址随后决定这次访问触及哪些扇区、由哪一级存储供数 |
| 同步操作数 | 操作指向哪个 barrier 对象（其共享地址）、到达数、期待的事务字节数、等待的相位奇偶、多播与远端目标的逐 lane cluster rank（`%getctarank`） |
| TMA 搬运 | tensor map 描述符镜像与坐标，二者给出被搬动的内存块 |
| barrier 轮询 | `PHASECHK`/`TRYWAIT` 类轮询各带一条 AFTER 记录，记下它观测到的 ready 掩码 |
| 全局存储同步 | 线程块在每次用于跨 CTA 信号的强 load、store、atomic 或 reduction 前后读写的值字 |
| cluster 成员 | 每个 CTA：它在 cluster 内的 rank 与 cluster 的形状，从 `%cluster*` 寄存器读出 |
| launch 事实 | grid/block/cluster 维度、设备，以及 CTA→SM 放置（来自 placement 运行） |

运行的两个结果是直接采到的，而不是推导出来的：一次轮询是否观测到 barrier ready，以及同步变量取过的值。关于"何时"的其余一切由还原给出。

## 2. 不采什么，为什么

- **时间。** 插桩会拖慢 kernel，被插桩运行的时序不是真实运行的时序。trace 回答"是什么"；静态事实与机制事实回答"何时"。
- **轮询自旋的次数（作为时序）。** 对同一个 barrier 的一串失败轮询折叠成一次等待，长度由引擎决定。折叠发生在转换阶段（`cpp/capture/poll_folding.cpp`、`python/cyclescope/convert/poll_folding.py`）：同一（对象、奇偶）上的轮询序列，若最后一次采样报告 ready 且轮询之间的指令都没有副作用，就折叠成幸存的那次轮询。

## 3. tracer 如何决定记什么

tracer 是一个 NVBit 工具（`tools/tracer/`）：kernel 带着真实输入在真实 GPU 上运行，注入的代码在被插桩指令之前写记录。记录哪些操作数按指令形式决定，只依据二进制本身，经架构表判定（`tools/tracer/arch_table.h` 与 `arch_hopper.h`、`arch_blackwell.h`）：需要采操作数的 opcode 族、哪些 opcode 带 mbarrier 字或其伴随字、TMA 坐标寄存器组的位置、SASS 语法是否有描述符记号。不从 kernel、它的库或输入形状推断任何东西。

## 4. 静态事实

二进制提供编译器的决定。tracer 把它见到的每个函数的 SASS 转储出来（`functions/<id>.tsv` 加 `functions.jsonl`）：每条指令的操作、操作数文本与控制字——stall 计数、yield 提示、设置与等待的记分板、reuse 标志。转换把它解码成静态指令模型；运行时事实给出静态形式留空的值。

## 5. 采集目录

`cyclescope capture --suite <suite> --case <case> --out <dir>` 每个 case 留下一个目录：

| 文件 | 内容 |
|---|---|
| `run.json` | 这次运行：launch 选择、返回码、tracer 哈希、判定 |
| `run_env.json` | 记录下的环境（密钥已遮蔽） |
| `app.log` | 应用自身的输出 |
| `launches.jsonl` | 进程发起的全部 kernel launch |
| `functions.jsonl`、`functions/<id>.tsv` | 每个函数的静态 SASS |
| `tensor_maps.jsonl` | tracer 见到的 TMA tensor map 镜像 |
| `trace-<launch>.bin.zst` | 选中 launch 的记录流 |
| `placement/` | placement 运行自己的目录（同布局） |
| `trace_check.txt` | 完整性检查的报告 |

## 6. 选择 launch 与两次运行

suite case 用正则加出现次数指定一个 kernel。driver 解析 case（`python/cyclescope/capture/driver.py`、`launch_selection.py`），必要时用 ncu 对应用的 launch 做一次普查，然后选中一个 launch——绝不跟踪任意一个。随后是两次带插桩的运行：

- **trace 运行**，记录上面描述的指令流；
- **placement 运行**（`CYCLESCOPE_CAPTURE_MODE=placement`），一次更轻的采样，在函数入口与出口记录每个 warp 的 `%smid` 和 warp 槽位——仿真需要的 CTA→SM 放置。

硬件参照是单独一次不带插桩的运行：`cyclescope reference` 用 ncu（`--cache-control all`）给这个 launch 计时，使参照时钟与冷缓存条件本身也成为记入作业的事实。

## 7. 代码位置

| 部分 | 位置 |
|---|---|
| NVBit 工具、记录生成 | `tools/tracer/tracer.cu`、`inject.cu`、`record_format.h` |
| 按架构的采集表 | `tools/tracer/arch_table.h`、`arch_hopper.h`、`arch_blackwell.h` |
| TMA 描述符操作数 | `tools/tracer/tensor_map_operand.h` |
| 采集 driver、launch 选择、完整性检查 | `python/cyclescope/capture/{driver,launch_selection,ncu_tools,trace_check}.py` |
| 硬件参照 | `python/cyclescope/reference/ncu_runner.py` |

driver 设置的 tracer 环境变量（另有一些诊断变量，见 `tracer.cu`）：

| 变量 | 作用 |
|---|---|
| `CYCLESCOPE_TRACE_DIR` | 记录流的输出目录 |
| `CYCLESCOPE_CAPTURE_LAUNCHES` | 跟踪哪些 launch（`<kernel>@<occurrence>`） |
| `CYCLESCOPE_CAPTURE_CLUSTERS` | 跟踪哪些 cluster（`all` 或 id 列表） |
| `CYCLESCOPE_CAPTURE_MODE` | `placement` 表示 placement 运行 |
| `CUDA_INJECTION64_PATH` | 把 tracer 装入 CUDA 进程 |
