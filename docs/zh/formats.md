# 二进制作业格式

[English](../en/formats.md)

管线读写的二进制文件布局：`GSIMSEG`（段）、`GSIMJOB`（容器）、`GSIMRES`（结果）、`GSIMJNL`（发射日志）、`GSIMWLG`（等待账本）。只有需要解析或对比这些文件时才用本页——其他地方都链接到这里而不重复布局。所有整数小端；记录宽度在各文件的表目录里声明，绝不硬编码偏移。各表是做什么的见 [design/job-format.md](design/job-format.md)。

## 1. 文件外壳

```
FileHeader (72 B)
  magic[8]          "GSIMSEG\0" | "GSIMJOB\0" | "GSIMRES\0" | "GSIMJNL\0" | "GSIMWLG\0"
  u32  format_version = 1
  u32  header_bytes = 72
  u64  file_bytes
  u64  directory_offset
  u32  table_count
  u32  flags            bit0 = payload sha256 present
  u8[32] payload_sha256  （该字段清零后对整文件求 hash）
TableDirectory: table_count x TableEntry (32 B)
  u32 tag (4 字符 ASCII) | u32 record_bytes | u64 record_count |
  u64 offset (8 B 对齐) | u64 reserved = 0
```

- `record_bytes = 0` 表示字节 blob（STRS）；此时 `record_count` 是字节长度。未知 tag 忽略。`u32 0xFFFFFFFF` 是唯一的 "none" 哨兵（`u64` 字段用 `0xFFFFFFFFFFFFFFFF`）。

## 2. 段 `GSIMSEG`（一个 cluster）

**版本 2（拆分程序表）**：`format_version = 2`。七张共享程序表（`PROG`、`NODE`、`GRD `、`LEAF`、`PARM`、`PTS `、`AXIS`）不在段里；每个 segments 目录一份 `segments/program.tables`（`GSIMPRG0` magic 的文件，恰好装这几张表）。读取侧拼回；META `program_digest` 仍指共享表。C++（`cyclescope-convert`）与 Python（`program_tables_payload`）写出逐字节相同的 `program.tables`。版本 1 仍可读。

| tag | 记录 | 字段（字节） |
|---|---|---|
| META | 120 | tree_digest[32], program_digest[32], converter_digest[32], cluster_id, cta_count, warp_count u32, instruction_count u64, flags u32 (bit0 model_complete) |
| CTAS | 56 | cta_index, cta_x/y/z, cluster_rank, cluster_shape_x/y/z, sm_id, peer_sm_id, multicast_mask u32; cta_group, is_pair_leader u8; tmem_columns u16; first_warp, warp_count u32 |
| WARP | 32 | cta_index, warp_in_cta, warp_slot, smsp u32; first_instance, instance_count u64 |
| TMPL | 68 | operation_class u32 (STRS 偏移), flags u32（arch 规则的 `instruction_flags`：bit0 memory、1 atomic、2 barrier、3 branch、4 async、5 tensor、6 predicated、7 tmem（Blackwell）、8 clc（Blackwell）、9 DEFER_BLOCKING：opcode 文本带 `DEFER_BLOCKING` 修饰符的 `BAR`，Python 与 C++ 转换器都会置位；operation class 仍为 `BAR`；仅 stall 账本的规则 B 读取）, resource u32, issue_group_resource u32; source_begin u64, source_count u32, destination_begin u64, destination_count u32; scoreboard_read/write/wait_mask u8, control u8 (bit0-3 stall, bit4 yield); timing_begin u64, timing_count u32, semantic_kind u32, semantic_u64 u64 |
| INST | 16 | template, semantics, memory_access, code_line u32 |
| REGS | 4 | register u32: bits 0-23 寄存器 id（命名空间：R<n>→n，UR<n>→256+n，P<n>→512+n，UP<n>→768+n，B<n>→1024+n；RZ/URZ/PT/UPT/SR_* 等硬连线或特殊名不发行、不进模板键），bits 24-31 flags（bit0 reuse_served：pc-16 的静态指令在同一操作数槽带 reuse 标志且同名寄存器；bit2 reuse_carrier：本指令自己控制字 reuse_mask 在该源槽的位；bit1 pair：64 位源操作数（`.64`/`.F32x2` 后缀，或 DADD/DMUL/DFMA/DSETP/DMNMX 类，或 FFMA2/FADD2/FMUL2 类中不带 `.F32` 后缀的 token——`.F32` 广播标量不是对）；目的寄存器行 flags=0。源槽序 = 操作数文本去掉目的寄存器与其后的普通谓词目的（PT/UPT/P<n>/UP<n>）后的顺序，非寄存器 token 也占槽位；与 reuse 位逐位对应） |
| SEMA | 60 | kind u16, flags u16, earliest_issue u64, field[6] u64 (§5)；mbarrier_wait：flags 0x200 = wait origin，此时 `earliest_issue` 存该幸存轮询所代表轮询序列首个 waiter 的绝对代码地址（代码地址表空间；可选 uniform 操作数事实 `wait_origin` 由轮询折叠在首个被折叠轮询地址与幸存轮询不同的时候写入）；引擎只为 stall 账本读取它；没有该事实即无 origin，facts 标记不变 |
| MACC | 48 | sector_begin u64, sector_count u32, flags u32 (低 3 位 = space: 1 shared, 3 constant; bit5 remote; bits 6-8 = access_kind: 0x0 load / 0x40 store / 0x80 atomic / 0xC0 async_copy / 0x100 async_reduce / 0x140 prefetch; bit9 = 不带 .BYPASS 的 LDGSTS 填 L1)；facts 空间 `shared_cluster` 解析为 1——引用带远端 rank 时置 bit5——`global_system` 为 0，`param` 为 4), sequence u64, shape u64, timing_begin u64, timing_count u32, cache_hint u32 (位域，Blackwell L1/L2 提示) |
| MSEC | 16 | address u64, byte_mask u32, lane_mask u32（扇区行只在一次访问内按 lane 合并；不同访问即使同扇区也各自有行） |
| TIME | 40 | kind, selector_a, selector_b, ready_warps u32; selector_c u64; cycles f64; enabled u8, pad[7] |
| PROG | 32 | mechanism u32 (STRS), event u8, schema_id u16, timing_kind u8, root_node u32, field_mask u64, operation_class_begin u64 (STRS), operation_class_count u32 |
| NODE | 24 | parent, child_begin, child_count, guard_begin, guard_count, leaf u32 |
| GRD  | 24 | axis u32, operator u8 (0 eq / 1 range), pad[3], low u64, high u64 |
| LEAF | 8 | parameter u32 (首个 PARM 行), status u8 (0 measured/1 experimental/2 placeholder/3 unsupported/4 structural), pad[3] |
| PARM | 48 | schema_id u16, field_id u16, model_kind u8 (0 constant/1 discrete/2 linear/3 affine), pad[3], axis u32, value f64, slope f64, intercept f64, points_begin u64, points_count u32. discrete: value = placeholder_default，slope = 轴数 width，intercept ≠ 0 = fail_closed（查表不中抛 `unsupported:engine:discrete_miss`；无 placeholder_default 的叶构建器自动置 1） |
| PTS  | 16 | axis_value f64, value f64. constant 不用；带点的 linear 存 (x,y) 对；discrete 每点先存 `width` 行轴值再存一行值 |
| FACT | 12 | axis u32, value u64 (enum 轴存值下标) |
| USES | 38 | node u32, target u8, pad[3], target_index u64, schema_id u16, field_mask u64, fact_begin u64, fact_count u32 |
| AXIS | 4 | name u32 (STRS 偏移)；行号 = GRD/PARM/init-FACT 用的数字轴 id |
| SPLT | 16 | semantics, rank, credit, pad u32 |
| MBPI | 24 | cta_index, object u32, address u64, arrival_count u32, pad u32 |
| MBOL | 4 | object u32 |
| MBOB（可选） | 12 | cta_index u32, address u64 — 每个段内对象一行，给出归属 CTA 与 CTA 内 shared 地址（引擎 3-3 多播映射） |
| GSYN（可选） | 40 | instance u32 (段内), lane, kind, class u8, flags u8, address, after, before, sequence u64 — 字段语义同容器 GSYN 表减 segment 列；只在有观测时写 |
| CODE | 16 | instance u32, pad u32, code_address u64 |
| CLKM | 16 | warp, kind u32, instance u64 |
| STRS | blob | NUL 结尾字符串 |

USE targets：0 InstructionTemplate, 1 TimingRow, 2 InstanceTiming,
3 MemoryAccess, 4 Resource, 5 MemoryLevel, 6 GpuMemory, 7 Scheduler,
8 Instance, 15 none。

## 3. 程序表 `GSIMPRG01`（一棵树）

`cyclescope tree compile` 产物：转换器不重走 JSON 树所需的编译机制程序表。同 §1 文件外壳（magic `GSIMPRG0`）。表：

| tag | 记录 / blob | 内容 |
|---|---|---|
| META | 64 B blob | tree_digest[32] + program_digest[32] |
| PROG | §2 PROG 行（pad 字节 = timing_kind） | 机制程序，按名排序 |
| NODE / GRD  / LEAF / PARM / PTS  / FACT / USES / AXIS / STRS | 同 §2 记录布局 | 遍历输入，含 init 路径 FACT/USES |
| SCHM | 4 B: schema_id u16, target u8, pad u8 | 参数 schema id -> USES target（schema_registry.json） |
| AXV  | 16 B: kind u32 (1 enum), owner u8 (1 engine), pad u8, pad[2], values_begin u32, values_count u32 | 轴 kind/owner + enum 值窗口 |
| AXVN | blob | NUL 分隔的 enum 值名，按 AXV.values_begin 索引 |
| RAX  | 8 B: begin, count u32 | 每 NODE 行 -> RAXV 区间 |
| RAXV | 4 B: axis u32 | 每节点所需 capture 轴（叶模型轴的后序并集） |

消费方：C++ 段构建器 `cyclescope-convert` 经 `--programs` 读取；Python 构建器在内存持有同一组 payload。

## 4. 容器 `GSIMJOB`（一次 launch）

| tag | 记录 | 字段 |
|---|---|---|
| LNCH | 104 | grid_x/y/z, block_x/y/z, cluster_x/y/z u32 x9; shared_static_bytes, shared_dynamic_bytes, registers_per_thread, sm_count u32; l2_bytes u64; occupancy_blocks_per_sm, occupancy_active_clusters, function_prior_launches, reference_sm_clock_khz u32; reference_cache_control u8 (0 unknown/1 none/2 all；`cyclescope build` 只接受 2：0 -> `missing_fact:reference_cache_control`，1 -> `unsupported:reference_cache_control:none`), device_facts_present u8, pad u16; flags u32 (bit0 has_unplaced, bit1 has_global_sync, bit2 has_clc, bit3 partial_launch), native_ctas_launched u32; device char[16] |
| SEGD | 64 | segment, cluster_id u32, byte_offset, byte_length u64, cta_base, cta_count u32, segment_sha256[32] |
| PLCE | 28 | cta, segment, cta_in_segment u32, placement u8 (0 unplaced/1 placed), pad[3], sm_id, warp_slot_base u32, source u8 (1 trace/2 probe), pad[3] |
| GSYN | 44 | segment（该段在 SEGD 的序号；行由 `cyclescope build` 从各段可选 GSYN 表收集）, instance u32（段内）, lane, kind, class u8 (1 atom/2 red/3 ld_strong/4 st_strong), flags u8 (bit0 after/bit1 before 有效), address, after, before, sequence u64。before = pre_value（RED，tracer phase-0 尾字）或返回的旧值（ATOM，phase-1 字）；after = post(op, before, 操作数)（RED/ATOM：ADD/EXCH/CAS/MIN/MAX/INC/DEC/AND/OR/XOR、float ADD）、存储的操作数（ST）或目的字（LD）；0 pre_value 是已知值 |
| TOPO | 24 | sm, placement_domain, l2_partition, tpc_pair, gpc, pad u32（`gpc` 是 `memory.l2_merge.domain_gpc` 所选请求种类的 L2 读合并域） |

容器没有 L2 足迹表（原 `L2FP`/`L2FM` 随冷启动参照条件删除）：引擎每次 launch 从空缓存起跑。

段映像：每个完整 `GSIMSEG` 文件按 8 字节对齐附在表后，经 `SEGD.byte_offset/byte_length` 定位。

合并规则（§4）：CTA 身份——段的 `CTAS/WARP/MBPI/MBOB.cta_index` 带 launch 级线性 CTA id（`x + y*gx + z*gx*gy`）；合并后的作业把 CTA 按 CTAS 表序密排编号（`cta_index` == 合并 CTAS 的行号），改写引用字段并把容器 `PLCE.cta`（launch 级）翻译成密排 id，丢弃作业中不存在的 CTA 的行（partial launches）。`WARP.first_instance`、`CTAS.first_warp`、`INST.{template,semantics,memory_access}`、`TMPL.{source,destination,timing}_begin`、`MACC.{sector,timing}_begin`、`SPLT.semantics`、`MBPI.cta_index`、`CODE.instance`、`CLKM.{warp,instance}` 按前面各段的行数累加重定基。target 为每实例（0/1/2/3/8）的 USES 行逐段保留、`target_index` 重定基；全局 target 的 USES 与 PROG/NODE/GRD/LEAF/PARM/PTS/AXIS 组取第 0 段，并核对各段 META digest 相同。STRS 与 FACT 在每段以共享程序表前缀开头并带段内尾部（驻留的 opcode / 类字符串、每实例 FACT 行）：第 0 段全取；其后各段复用最长公共 STRS 前缀（NUL 边界）并追加尾部，FACT 前缀 = 其每实例 USES 行最小 `fact_begin` 之前的行（必须等于第 0 段的行）并追加尾部；该段的 `TMPL.operation_class`、尾部 `FACT.axis` 与 `USES.fact_begin` 随之重定基（跨界前缀的 FACT 区间是格式错误）。

## 5. 实例语义种类（SEMA.kind → field[0..5]）

字段名按 `python/cyclescope/job/semantics.py`。`generation` 沿用 mbarrier 编码：bit 63 = unresolved 标志，bit 0 = parity。`mbarrier_wait.field1` 打包 generation|parity；`object_list_begin|count` 在一个槽里打包高|低 u32；`wgmma.field1` 同样打包 accumulator_registers_begin|count。

| kind | 字段 |
|---|---|
| none | - |
| mbarrier_init | object, expected_arrivals, scope (0 cta/1 cluster), initial_phase (0/1, 默认 0；Hopper：SYNCS.EXCH.64 init 值高字 bit 31，Blackwell：由 init 值解码，转换器填), -, object_list_begin\|count（提升为 init 的逐 lane 存储初始化列表里每个对象，如 CUTLASS 每级 barrier 一条 STS.64 的 init） |
| mbarrier_arrive | object, generation, transaction_bytes, arrival_count, destination_mask, object_list_begin\|count。逐 lane 互异的 barrier 字（`mbarrier_words` 操作数事实；CLC 响应 arrive：lane i -> CTA i）在 CTA 内偏移共享时变成发出者对象上的 destination_mask + SPLT credit（每 rank lane 数），否则为 MBOL 列表。`SYNCS.*.RED` 字是 shared::cluster（无 tag = rank 0）；非 RED 的 SYNCS 字是发出者自己的。flags bit8 (0x100) = `.A<n>TX` complete_tx：`arrival_count` 0 且 `transaction_bytes` 完成期望 tx 而非到达/期望（CLC 未用时 nvjet 以此提交 16 字节 CLC 响应；`.A<n>TR` 仍是 expect_tx） |
| mbarrier_expect_tx | object, generation, transaction_bytes, scope |
| mbarrier_wait | object, generation\|parity, block_offset, scope, timeout_hint, after_ready_mask (flags bit0 = 已知) |
| mbarrier_inval | object |
| cta_barrier_arrive / cta_barrier_wait | barrier, expected_threads (0 = 整个 CTA), generation；BAR.SYNC / BAR.RED 是 cta_barrier_wait 且 flags bit1 = 同时到达。操作数：立即数原样；寄存器 id 是捕获值低 16 位、寄存器 id 旁的寄存器计数是高 16 位（`BAR.SYNC R0, R0` 打包 `(count << 16) \| id`），立即数 id 旁的寄存器计数是原始值；无计数 = 0：每个未退出的 CTA warp（引擎扣已退出 warp）；无捕获值的寄存器是 `missing_fact:cta_barrier_operand` |
| cluster_barrier_arrive / cluster_barrier_wait | generation, destination_mask |
| tma_load | object, generation, multicast_mask, shared_bytes, global_bytes, tensor_map。`shared_bytes` = box 字节（`tile::gather4` / `scatter4`：4 个 box 行）；flags bit7 (0x80) = `.2CTA` (cta_group::2)：引擎把每个目的 rank 的字节记到其 pair leader（rank & ~1）。TMA / bulk / UTCBAR / CLC 操作数的 mbarrier 字是 shared::cluster 地址，rank tag（bits 24+）为归属者，无 tag = rank 0 |
| bulk_group_issue / commit / wait | group, pending_allowed, bytes（仅 issue）；PTX `cp.async.bulk` 异步组：UTMASTG / UTMAREDG / S2G UBLKCP / UBLKRED 发射，UTMACMDFLUSH 与 UTMACCTL.COMMIT（两架构；UTMACCTL.IV 是 descriptor_visibility_acquire，MEMBAR.ALL.GPU 是 descriptor_visibility_publish）提交，DEPBAR.LE SB0,N 等待；有开放 issue 时 DEPBAR 隐含提交（SEMA.flags bit2 = 隐式 commit） |
| async_copy_issue / commit / wait | group, pending_allowed, object, arrival_count, transaction_count（仅 commit；LDGDEPBAR 上两个都 0；ARRIVES.LDGSTSBAR：.TRANSCNT → arrival_count 0 + transaction_count = lanes，否则 arrival_count = lanes） |
| bulk_copy / bulk_reduce | object, generation, bytes, multicast_mask |
| warpgroup_arrive / wait | group（每 warp 序号 / 已提交 unit 数）, pending_allowed（DEPBAR keep；立即数缺 -> missing_fact:warpgroup_wait_keep，非 gsb0 -> unsupported:warpgroup_scoreboard，keep>7 -> unsupported） |
| wgmma (hopper) | group, accumulator_registers_begin\|count, shape_packed (m<<32\|n<<16\|k), input_dtype, accumulator_dtype, source_mode (1 ss/2 rs)；折叠 opener -> flags GROUP_OPENER，gsb0 操作数 -> GROUP_COMMIT |
| umma_issue / umma_commit (blackwell) | issue：kind (1 f16/2 tf32/3 f8f6f4/4 mxf8f6f4/5 mxf4/6 mxf4nvf4), cta_group, a_source (1 smem/2 tmem), tensor_m\|tensor_n, descriptor (idesc 捕获字)；commit：object, cta_group, commit_mask, scope |
| tmem_alloc / dealloc / relinquish | columns（捕获 UR 按 32 列粒度映射；relinquish 0；dealloc 无捕获值 -> SEMA_UNKNOWN，引擎按 tcgen05.dealloc = 释放该 CTA 持有的分配求解；alloc 仍 missing_fact） |
| tmem_load / store | shape 枚举 (1 16x64b/2 16x128b/3 16x256b/4 32x32b, 5 16x32bx2 half (16dp32bit_t0_t15 / _t16_t31)；dp-bit 拼写映射到同一编码；裸 xN 与 xN.MAX.* = shape 4 + repeat N), repeat；LDTM.STAT -> flags STAT |
| tmem_copy | 仅 UTCCP：shape = (N<<16)\|M（由 Nx32dpMbit 或单块 <D>dp<M>bit 打包），repeat 1 |
| tmem_wait_store | （无字段；FENCE.VIEW.ASYNC.T） |
| clc_query (blackwell) | object, generation, destination_mask, response_identity, outcome |
| shared_store_async | object, bytes, destination_rank, object_list_begin\|object_list_count |
| global_sync | observation_kind (1 release/2 acquire/3 atomic), class (1 atom/2 red/3 ld_strong) — 每条带 GSYN 观测的 ld_strong / atom / red 都发此行（st_strong 只是生产者，不发）；引擎以本行为键做 acquire 等待与原子前的门，配对仍在引擎（GSYN 表） |
| reconvergence_start / complete | region, envelope_cycles |
| nanosleep | requested_ns；flags bit6 (0x40) = `.SYNCS`（等待集 = 其轮询区——两次非失败 mbarrier wait 之间——内全部已知失败轮询对象，外加区尾 wait 的 (object, parity)；任一已释放 -> 不睡；睡眠时长按 PTX ISA 上限 1 ms） |
| descriptor_visibility | operation (1 publish/3 error_check/4 acquire/5 proxy_fence；store 只登记候选，不发行), descriptor_count, descriptor_address_0..3 |
| clock_marker | kind |
| dependents_launch | （无字段；PREEXIT 执行 = griddepcontrol.launch_dependents；引擎在 CTA 完成时向分派串行器计费） |

### 5.1 字段来源

| 字段 | 来源 |
|---|---|
| object | 取自该指令访存引用 lane 地址的 mbarrier shared 地址，否则显式 `mbarrier_address` 操作数事实（Blackwell UTMALDG.* 由捕获 UR 值按编码的 object id 机械恢复）；`(cta_index, address)` → 对象下标；无对象打包为 `SEMA_NO_OBJECT` = 0xFFFFFFFF（job-format §2 "none" 哨兵），解包回 None |
| expected_arrivals / arrival_count / expected_threads | `barrier_operands` 立即数 token（寄存器 token → 0，待操作数值侧车） |
| transaction_bytes | arrive.expect_tx 操作数（`opN` 立即数事实） |
| parity | `mbarrier_phase_operand.value`（bundle payload 事实） |
| generation | 恒 `GENERATION_UNRESOLVED`（bit63）；相位由引擎求解 |
| after_ready_mask | `after_ready_mask` 操作数事实；存在时置 SEMA.flags bit0 |
| destination_mask / multicast_mask | 逐 lane `getctarank` rank 事实按位或 |
| barrier | 命名 barrier id 操作数（`op0` 立即数） |
| group | 每 warp 顺序异步组计数；commit 关组 |
| pending_allowed | wait 立即数（`DEPBAR`/wait_group 操作数） |
| bytes | 指令大小操作数（`opN` 立即数），否则 TMA box 字节 |
| tensor_map | tensor-map 事实 id；shared/global_bytes = box_dim 积 × element_bytes |
| descriptor operation/count/address_N | publish：按基址排序的完整 128 字节窗口（8 个 16B STG 块），count=N（0 -> 无行，>4 unsupported）；acquire：单基址操作数事实；error_check/proxy_fence：count=0 |
| pending_allowed / depbar_keep | `DEPBAR.LE SB0,N` 尾立即数 token（barrier_operands / operands）；只有 SB0 FIFO 计数，N<=7 恒可，N<=63 需同 warp 内先前 LDGDEPBAR/bulk-flush 证据 |
| requested_ns | NANOSLEEP 立即数 |
| scope | 0 = CTA，有远端 shared 寻址 / rank 事实时 1 = cluster |
| object_list (MBOL) | 发散 arrive 的逐活跃 lane 互异 mbarrier 地址 |
| missing facts | 解码时 `missing_fact:<operand> (<opcode> @pc)`；bundle 缺值时构建器发 `INST.semantics = 0xFFFFFFFF` 并计数 |

## 6. 时间行种类（TIME.kind）

一个机制程序对应一个 kind。

| id | 名称 | 机制 | 使用方 |
|---|---|---|---|
| 1 | `global_load_dependency` | — | — |
| 2 | `global_store_acceptance` | — | — |
| 3 | `shared_store_visibility` | `storage.shared.visibility` | 两者 |
| 4 | `global_store_visibility` | — | — |
| 5 | `shared_issue_interval_smsp` | `storage.shared.issue` | 两者 |
| 6 | `shared_store_acceptance_sm` | `storage.shared.acceptance_sm` | 两者 |
| 7 | `dependency_result_smsp` | — | — |
| 9 | `tma_verified_inflight_domain` | — | — |
| 10 | `tma_overlap_operation` | — | — |
| 13 | `hopper_wgmma_handoff_consumer` | `hopper.wgmma.accumulator_handoff.consumer` | Hopper |
| 15 | `atomic_issue_interval_sm` | `storage.atomic.issue` | 两者 |
| 16 | `nanosleep_warp_delay` | `control.nanosleep` | 两者 |
| 17 | `cta_barrier_release` | `synchronization.cta.release` | 两者 |
| 18 | `async_copy_mbarrier_arrival` | `synchronization.cp_async.arrival` | 两者 |
| 19 | `tensor_map_descriptor_visibility` | `synchronization.descriptor.visibility` | 两者 |
| 20 | `lifecycle_contract` | — | — |
| 21 | `service_start_interval` | `storage.async.service_start` | 两者 |
| 22 | `mbarrier_phase_release` | `synchronization.mbarrier.release` | 两者 |
| 29 | `hopper_qgmma_fmnmx_handoff_consumer` | `hopper.wgmma.fp8_fmnmx_handoff.consumer` | Hopper |
| 31 | `tma_overlap_operation_signature` | — | — |
| 32 | `tma_store_background_drain_extension` | — | — |
| 38 | `tma_operation_completion` | `storage.tma.completion` | 两者 |
| 39 | `storage_ldgsts_group_completion` | `storage.ldgsts.completion` | 两者 |
| 42 | `cluster_barrier_release` | `synchronization.cluster.release` | 两者 |
| 43 | `hopper_tma_descriptor_refetch` | `storage.tma.descriptor_refetch` | 两者 |
| 44 | `hopper_wgmma_register_file_issue_steal` | `hopper.wgmma.register_file_issue_steal` | Hopper |
| 51 | `blackwell_umma_commit` | `blackwell.umma.commit` | Blackwell |
| 52 | `blackwell_tmem_load` | `blackwell.tmem.load` | Blackwell |
| 53 | `blackwell_tmem_store` | `blackwell.tmem.store` | Blackwell |
| 54 | `blackwell_tmem_alloc` | `blackwell.tmem.alloc` | Blackwell |
| 55 | `blackwell_tmem_copy` | `blackwell.tmem.copy` | Blackwell |
| 56 | `blackwell_clc_query` | `blackwell.clc.query` | Blackwell |
| 57 | `blackwell_tmem_load_stat` | `blackwell.tmem.load_stat` | Blackwell |
| 59 | `blackwell_umma_pair_arrive_skew` | `blackwell.umma.commit.pair_skew` | Blackwell |
| 61 | `mbarrier_trywait_timeout` | `synchronization.mbarrier.trywait_timeout` | 两者 |
| 63 | `tensor_pipe_beat` | `tensor.pipe` | 两者 |
| 64 | `tensor_result_latency` | `tensor.completion` | 两者 |
| 65 | `global_store_issue_interval_smsp` | `storage.global.store_issue` | 两者 |
| 66 | `shared_fence_drain` | `storage.shared.fence` | 两者 |
| 67 | `warp_queue_entry` | `instruction.warp_queue` | 两者 |
| 80 | `shared_store_acceptance_smsp` | `storage.shared.acceptance_smsp` | 两者 |
| 81 | `tma_reduce_store` | `storage.tma.reduce` | 两者 |

列表来源：`python/cyclescope/job/timing_rows.py` 中的 `TimingKind`。标 "—" 的 kind 在枚举里有定义，但随发的树里没有程序使用它。

## 7. 结果 `GSIMRES` / `GSIMJNL` / `GSIMWLG`

RSLT（一行）：cycles（任一 SM 活跃周期数的最大值——其上 warp 驻留区间
[entry, done) 的并集长度；原先的「末次完成时刻 + envelope」值为
TELE group 2 `last_completion_cycle`）, issued, completed, hard_constraint_failures u64,
flags u32。SMSP：sm, smsp u32, first_active, last_active u64。WAIT：
scope, cause_kind u32, cycles u64。RWAIT：resource, cause_kind u32,
cycles u64。TELE：group, counter u32, value u64。
JRNL（GSIMJNL）：warp, smsp, instance u32, issue_cycle u64,
operation_class, template u32。WLDG（GSIMWLG）：warp, cause_kind u32,
cycles u64, count u32。

### Facts 文件（`GSIMCAP`）

`convert --capture` 以同一外壳写逐 cluster 的 facts 文件。其中一张值得知道的表：`MREX`，每访存引用一条扩展行 `transfer_bytes, addr_begin, addr_count, cut_rows`（20 B；`cut_rows` 是 tiled TMA load 第 0 维 box 部分越界时的界内行数，否则 0/-1）。记录宽不同的 MREX 表拒收为 `missing_fact:mrex_record_bytes`。
