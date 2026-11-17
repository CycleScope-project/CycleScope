# FlashMLA marker 插桩方案（CS2R SR_CLOCKLO 口径统一）

- 任务：T0005 修订 3-②；作者：T0005gMarkers（2026-09-29）
- 定位：`suites/flashmla/`（suite 钉上游 ba89a3466e9470ad08ab39738d4e7bb66989e1e7，见 `upstream/PINNED.md`）
- 本文件按主调度协调约束先落外置根，随后由主调度迁入仓库 `suites/flashmla/harness/marker_timing_plan.md`。

## 0. 口径基准（必须对齐的合同）

- 模拟器口径：`docs/simulator.md` §2.5（:190-199）——硬件与模拟必须同测 **核内最外层 clock-marker pair**；模拟器在 `simulation.clock-marker-timing.v1` 下按 **CS2R `SR_CLOCKLO` 操作起点**采样，重算 `max(all_warp_end) − min(all_warp_start)`；无 marker 通道 → 显式失败 `simulate_failed_marker_window`。
- 296 硬件分母：`scripts/install/build_cluster_sm_296_hardware_rows.py:36-37` 固定 `MARKER_SCOPE="outermost_in_kernel_clock_marker_interval"`、`MARKER_METRIC="model.marker_timing.marker_interval_cycles"`；非 marker 证据只能带 `needs_marker_remeasure: true`，绝不入 `median_cycles`（同文件 :19-20）。
- 数值方法（近集同款）：`suites/flashattention3/harness/marker_timing.py:9-17`——每 (cta,warp) 一行 `[start_clock64, end_clock64, start_globaltimer, end_globaltimer, smid, cluster_packed, cluster_ctarank]`；host 侧 `cycles = (max(end_timer)−min(start_timer)) × median(Δclock64/Δglobaltimer)`，行有效性 `row[0]>0 ∧ row[1]>row[0] ∧ row[2]>0 ∧ row[3]>row[2]`；27 raw → 9 formal（`RAW_SPANS_PER_SAMPLE=3 × FORMAL_SAMPLES=9`，marker_timing.py:49-52）。
- 常量复用：`MARKER_SCOPE` / `MARKER_METHOD="device_kernel_clock64_over_globaltimer"`（marker_timing.py:44-45）。**globaltimer 裸跨度不得作为 `hardware_rows.median_cycles`**。

## 1. 现状（为什么必须插桩）

- harness 计时只有 whole-operation globaltimer：`harness/flashmla_cluster_sm.py:8`（"Whole-operation device timing uses the shared delivery globaltimer protocol"）、`:41` `from globaltimer_timer import measure_operation`；`harness/globaltimer_timer.py:1`（"Device-globaltimer timing for a possibly multi-kernel CUDA operation"）。核内无任何 clock64 marker 通道 → 追踪流里主核无 `CS2R SR_CLOCKLO` → 换算层只能给 `clock_markers: not_present` → 模拟必然 `simulate_failed_marker_window`；globaltimer 跨度口径不可比。
- `scripts/build_cluster_sm.sh:79-84` 只对计时库 `libdelivery_globaltimer_timer.so` 做 SASS 检查（SR_CLOCKLO/SR_GLOBALTIMERLO），与 kernel 无关。
- 一次 op 最多 3 个 kernel（flashmla_cluster_sm.py:5-9）：
  1. `get_mla_metadata_kernel`（仅首调，排程元数据）— `upstream/csrc/kernels/smxx/decode/get_decoding_sched_meta/get_decoding_sched_meta.cu`
  2. **主核** `flash_fwd_splitkv_mla_kernel(__grid_constant__ const DenseAttnDecodeParams, __grid_constant__ const TmaParams)` — `upstream/csrc/kernels/sm90/decode/dense/splitkv_mla.cuh:958-960`，grid `(num_m_blocks, num_kv_heads, num_sm_parts)`；host 侧 `cudaLaunchKernelEx` + PDL（splitkv_mla.cuh:1338-1351，:1224-1225 注释确认 combine 经 Programmatic Dependent Launch 衔接）
  3. `flash_fwd_mla_combine_kernel`（num_splits>1 时）— `upstream/csrc/kernels/smxx/decode/combine/combine.cu`

## 2. kernel 侧最小改动设计（FA3 `--marker-timing` 同款，measurement-only）

参照物：CUTLASS/FA3 共用的 override 机制 `suites/cutlass/harness/override/cutlass/device_kernel.h:95-96,136-137`（kernel 首尾 `mov.u64 %0, %%clock64 / %%globaltimer`，写 per-(cta,warp) 行，"The runner sets the device pointer before launching the kernel"）与 FA3 封装 `suites/flashattention3/harness/marker_timing.py`。

### 2.1 新增 marker 头（不进上游语义）

新文件 `suites/flashmla/harness/override/flashmla_marker.h`：

```c
#ifdef FLASHMLA_DELIVERY_MARKER_TIMING
#define FLASHMLA_MARKER_MAX_ROWS (1u<<18)          // 262144 行覆盖最大 grid
#define FLASHMLA_MARKER_FIELDS 7                    // 与 FA3/CUTLASS 布局逐字段一致
__device__ unsigned long long* flashmla_marker_main_ptr;     // splitkv 主核区
__device__ unsigned long long* flashmla_marker_combine_ptr;  // combine 区（次要证据）
extern "C" void flash_mla_delivery_marker_set(const void* main_ptr, const void* combine_ptr);
__device__ __forceinline__ void flashmla_marker_read(unsigned long long* clock_out,
                                                     unsigned long long* timer_out) {
    asm volatile("mov.u64 %0, %%clock64;"     : "=l"(*clock_out) :: "memory");
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(*timer_out) :: "memory");
}
#endif
```

- 三个 kernel 编译进同一 `flash_mla_cuda` 扩展 .so，`extern "C"` setter 一并编译其中；harness 用 `ctypes.CDLL(<flash_mla_cuda .so>).flash_mla_delivery_marker_set(...)` 在**每次正式计时前**设指针（语义同 CUTLASS "runner sets the device pointer"，无需改任何 kernel 签名/ABI，`__grid_constant__` 参数原样）。

### 2.2 主核插桩点（splitkv_mla.cuh，仅 2 处 + 头包含）

- 入口：`:960` 函数体最前（`m_block_idx/k_head_idx/partition_idx` 取号后、任何 TMA prologue 前）读一次 `(clock,timer)`；
- 出口：kernel 唯一返回路径末尾（最后一次 barrier/store 之后、`}` 之前）再读一次，并写行：

```c
row = ((blockIdx.x + gridDim.x*blockIdx.y) + (gridDim.x*gridDim.y)*blockIdx.z)
      * (blockDim.x/32) + (threadIdx.x/32);            // 12 warps（NUM_THREADS=384）
unsigned long long* base = flashmla_marker_main_ptr;
if (base) {  // row < FLASHMLA_MARKER_MAX_ROWS
    unsigned long long* r = base + row*FLASHMLA_MARKER_FIELDS;
    r[0]=start_clock; r[1]=end_clock; r[2]=start_timer; r[3]=end_timer;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(r[4]));
    // r[5]=cluster_dim_packed, r[6]=cluster_ctarank（%%cluster_nctaid/%%cluster_ctarank，
    //   写法照抄 cutlass device_kernel.h:98-99 与 :136-137 一段）
}
```

- PDL 注意：主核 `cudaTriggerProgrammaticLaunchCompletion`（splitkv_mla.cuh:1224 附近）允许 combine 提前启动，但 marker 行写在**各 kernel 自己的返回前**，跨核重叠只影响 whole-op globaltimer，不影响 per-kernel 行；入口读放在 prologue 前、出口读放在 epilogue 后即「最外层」。
- combine/sched-meta kernel 同法写各自区域（combine.cu、get_decoding_sched_meta.cu 各 2 处）；仅作次要证据，不进 formal 中位数。

### 2.3 编译开关

`scripts/build_cluster_sm.sh` 的 pip 源码构建步骤加 `-DFLASHMLA_DELIVERY_MARKER_TIMING`（默认关：无宏时头文件全空、上游语义逐字节不变；`setup.py` 的 `CXXFLAGS`/nvcc extra flags 注入点由 build 脚本 env 提供）。SM90 专用：本 suite 只编 sm90a gencode（build_cluster_sm.sh:4 注释），无跨架构顾虑。

## 3. host 侧（harness）改动

1. 新文件 `harness/marker_timing.py`：逐字移植 FA3 `marker_timing.py` 的缓冲布局（3 块 torch int64 张量承载 u64 位型，torch 无 uint64，marker_timing.py:19-22）、`marker_timing_requested`（env `FLASHMLA_MARKER_TIMING=1` 或 per-case `marker_timing` 键，marker_timing.py:57-60）、reduction（timer_span×median-ratio、行有效性、27→9）、常量 `MARKER_SCOPE/MARKER_METHOD` 原值。
2. `harness/flashmla_cluster_sm.py`：`measure_operation` 计时环改挂 marker 模式——每 raw launch 前 zero buffers → 设指针 → 调 `flash_mla_with_kvcache` → 读回主核区行；formal 行取**主核区**；combine/sched-meta 区落盘为次要字段（`marker_timing.combine_*`）。
3. 时长匹配护栏照抄 FA3：marker cycles 与 CUDA-activity 主核时长相对差 ≤10%（marker_timing.py:53 `DURATION_MATCH_RELATIVE=0.10`）。

## 4. 静态与捕获侧审计（防「通道静默失效」）

- 构建期 SASS 审计：build 末尾对 `site-packages-flashmla` 下 flash_mla 扩展 .so `cuobjdump --dump-sass`，抽 `flash_fwd_splitkv_mla_kernel` 文本断言 `SR_CLOCKLO≥2 且 SR_GLOBALTIMERLO≥2`（判据同 DG 侧 `text_value.count("SR_CLOCK")` 门，`tools/microbench/deepgemm_sm90_runner.py:1070-1071` 与 `deepgemm_attention_runner.py:435-438`；替换现在只查 timer .so 的 build_cluster_sm.sh:82-84）。
- 捕获期动态审计：镜像 FA3 `dynamic_clock_marker_audit`（`suites/flashattention3/harness/capture_cluster_sm.py:131-176`：追踪窗内**每个 warp 都执行了静态最外层 marker pair**，否则 CaptureRejected）与 marker-PC 选择（`suites/cutlass/harness/capture_cluster_sm.py:438-443`：`opcode==CS2R ∧ "SR_CLOCKLO" ∈ operands`，`function_name==flash_fwd_splitkv_mla_kernel`，`len(marker_pcs)≥2`）。捕获脚本已有 `ACTIVE_FROM_START=1` 窗口（`scripts/capture_cluster_sm.sh:65`），无需再动。

## 5. 上游源码可改性评估

- **可改，成本低**。`upstream/` 是钉版本的纯工作树导出（`PINNED.md`：`.git/` 已去、树内含 csrc/kernels），`scripts/build_cluster_sm.sh:24-46` 直接从该树 pip 构建（git submodule 缺失仅警告）；补丁只碰 `splitkv_mla.cuh`（2 处）、`combine.cu`（2 处）、`get_decoding_sched_meta.cu`（2 处）、`params.h` 不动、kernel 签名/ABI 不动、默认无宏不编。
- 与 43cf2a34→修订 3-① 的历史重写无冲突：补丁不触及 `csrc/cutlass`（该子模块 147f5673 改为构建期取）。
- 合规要求：`PINNED.md` 追加「delivery measurement-only patch」记录（补丁文件清单 + 补丁后树 sha256），保持上游钉哈希与补丁哈希分开登记；`upstream_sha` 语义不变。
- 工作量估计：kernel 侧 ~60 行、harness 移植 ~200 行、build/审计钩 ~30 行；nvcc 编译无需 GPU（sm90a gencode 交叉编译），GPU 仅用于冒烟与两轮 formal。

## 6. 验收清单

1. 无宏构建：扩展 .so SASS 与上游基线逐位一致（回归零风险）。
2. 带宏构建：SASS 审计过（§4 构建期）；单例冒烟 marker cycles 与 whole-op globaltimer/活动时长比合理（clock64/globaltimer 比值 1.8–2.0 参考带，`suites/README.md:223-224`）。
3. 捕获一例：bundle `clock_markers: present`，换算 `marker_interval_cycles` 非空；模拟走 `simulation.clock-marker-timing.v1`，不再出现 `simulate_failed_marker_window`。
4. 16 例两轮：`hardware_rows.json` 行 `scope=outermost_in_kernel_clock_marker_interval`、`comparison_kind=marker_interval`，与 296 schema 同构。
