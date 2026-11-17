# DeepGEMM 裸测实际加载与 unit 分母

这是独立于原 formal whole-operation timing 的 host-only 证据通道。旧86七字段
raw、14 across-marked-kernels scope、原 grid/warps/设备 MARKER_HEADER 和
historical eligibility 均不升级、不覆盖。NCU、host `.so` SHA 和 cache 唯一名字
不能证明本次 GPU Function。采集、编译和 SASS 导出必须在授权 H100 cluster
job 内执行（predict-only、GPU≤1）；artifact/build/cache 外置。
H100 image 的cuobjdump还需要独立nvdisasm。沿用T0016已验证的
`$HOPPER_ARTIFACT_ROOT/tasks/T0012/runner/nvdisasm`，在job内先验证版本，
将其目录放入PATH并导出NVDISASM_PATH；CUOBJDUMP采用明确的已验证工具路径，
actual导出记录实际argv/rc/SASS/stderr。工具缺失不能由原SASS或NCU替代。

## 实际 host 接点

`upstream/csrc/jit/handle.hpp` 仍用显式 `RTLD_LOCAL` libcuda handle；首次解析
真实 pfn 后调用共享 `fi_wrap_driver_symbol`，因此 DG 绕 LD_PRELOAD 的 launch
现在经过同一个 observer。≥12.4 使用原 LibraryLoadFromFile→EnumerateKernels→
KernelGetFunction，并以真实 LibraryGetModule 获取 module；旧 driver 分支仍为
ModuleLoad→ModuleGetFunction。两分支成功后 `fi_deepgemm_bind` 写真实
loaded_path/library/module/CUfunction/current-context/完整名字，unload 记录代际
结束。没有 cuKernelGetLibrary 或 cudaKernel_t→CUfunction 强转；runtime API
模式要求 actual 时明确拒收（不能伪造 opaque-handle 桥）。

共享 observer 的 driver wrappers 在真实调用返回后记录 enqueue_status，保留
完整名字、名字字节数、context/module/function、grid/block/effective cluster、
dynamic_smem、原stream及numAttrs/原cluster、PDL、cooperative属性值。DG所用
属性必须完整；其他未支持属性保留id并标明`config_complete=false`，不能签actual。
required-cluster 属性与 Ex cluster 配置核对，零维度不归一化为1，不改变实际 launch。
`fi_launch_cursor` 在同步后的 trial 窗口起止固定序号；窗口内每条 driver
launch必须有唯一有效的loaded Function/module/context绑定并成功enqueue，
否则整个trial拒收，不能因同窗口另一个kernel已匹配而静默丢掉缺module、
错context或失效绑定。原runtime名称仍只是诊断。`driver_identity=false`
不是选中目标，不能当成CUfunction，也不能用来证明同码。`marker_reset`
按cursor+thread接上的runtime launch是已证实的marker清零。其余runtime
launch留在窗口内，不丢弃；只有当它的名字等于选中Function、它的参与槽
等于选中unit的观测槽、或它的geometry无法与观测槽比较时，才拒收该选中
unit。同一loaded Function的两次enqueue若grid不同，先保留两条记录，再按
观测槽唯一归属选中unit；槽不能唯一归属，或选中名字对应多于一条物理
enqueue，则拒收该unit，不合并槽。窗口context检查只计driver launches。

## 构建与运行

原 package 的 `deep_gemm/sm90_suite_marker.json` 指向真实 build_log/source。
`prepare_deepgemm_sm90_package --host-only --marker-metadata ...` 只植入
handle.hpp host 接点，绝不调用再次插设备 marker 的 patch_source；所有 device
include 在前后 SHA 核对，并沿用原 full7 metadata。不同原/force package 分别
构建，不统一重写为另一个 package。

```sh
# 在获准的 H100 job 内，从稳定的新 source snapshot 执行：
python3 -m tools.microbench.prepare_deepgemm_sm90_package \
  --host-only --source ORIGINAL_BUILD_SOURCE --marker-metadata ORIGINAL_METADATA \
  --output EXTERNAL_PACKAGE --work-root EXTERNAL_BUILD
python3 suites/deepgemm/harness/run_case.py \
  --case-id ORIGINAL_CASE --package EXTERNAL_PACKAGE \
  --observer EXTERNAL_LIBACTUAL_OBSERVER_SO --work-dir NEW_EXTERNAL_RUN \
  --unit-json ORIGINAL_CAPTURE_UNIT_JSON
```

`run_case.py` 重放正式 ENV/CMD（不加 trace、不调用 profiler），保留正式原始
aggregation 分组。cache 仅用于隔离复制原输入清单；实际身份只来自 observer
真实 loaded_path，不扫描 cache 选择签名。实际 cubin/source 前后 SHA 必须
等于只读复制清单；JIT 新产物不在清单即拒收。复制前要求90GiB可用护栏。
为避免host包搬迁改变`-I`及DG compiler cache signature，runner经现有`_C.init`
API保留原formal package的device include根路径；先将其所有include SHA与
host-only metadata逐文件完全核对。新extension仅承载host观察接点，不改设备
root/flags或cache命名，仍只凭真实loaded_path与预先只读manifest验证实际cubin。

`bare_campaign.py` 接受原100 scope handoff，逐 case 分别保留 command/machine/rc、
loaded_code.jsonl、launches.jsonl、raw_marker_trials.jsonl、source_manifest.json、
实际 loaded cubin 的完整 SASS/stderr/rc 和 result/receipt。不重采NVBit99。

若传 `--unit-root`，每例原capture unit文件必须存在，缺失立即记录失败，
不退化为无unit测量。`--package-map`复用相同原package对应的已构建host包，但仅当
该包带当前观察接点（`host_package_is_current`：`actual_binding_host_hashes`
同时含`csrc/jit/handle.hpp`、`csrc/jit_kernels/impls/runtime_utils.hpp`、
`csrc/apis/attention.hpp`、`csrc/jit_kernels/impls/smxx_layout.hpp`）。
只含attention reset、缺layout transpose reset的r11包不是当前包；
缺接点/缺元数据/坏JSON的旧包一律剔除并从原源重建——不存在旧seed默认值或兼容shim；
`--skip-case-id`只用于拼接已完成bare原件，不改变100例原scope台账。
host-only wheel构建复用仓库CUDA wheel include/link closure（包含cusparse），
仅设置CPATH/LIBRARY_PATH与DG_FORCE_BUILD，不增加设备编译flag。

## unit JSON 与消费者合同

输入必须由原 capture 真实单位物化，不能按观测 valid rows 缩小：

- `type`、`kernel_file`、`ctas:[[x,y,z],...]`、原 `grid/block:{x,y,z}`；
- 原 `cluster:[x,y,z]`、完整 `kernel_name`；
- `clock_marker_policy:outermost_pair`；
- `kernel_sass_fingerprint:{method:cuobjdump_dump_sass_function_pc_words_v1,entry,sha256}`。

`actual_binding.py` 严格 JSONL 序号/尾行/完整名校验，按实际句柄/context/module/
pid/有效 load generation join，然后才核对名字与 capture 指纹。选中完整
Function 的 PC、SASS、两个编码 words 复用 campaign 指纹方法；重复 Function
section 或不完整编码拒收。每次 launch 必须原 geometry 完全一致，所有原
CTA/warp 七字段齐全且唯一；触及但 clock pair 损坏的槽也原样写 durable raw，
不会 silently filter。二维 marker 无法辨认的 geometry 不缩 grid，直接拒收。

新增 `timing.unit_denominator` 使用既有 `delivery.unit-marker-denominator.v1`；
clock64 为 per-SM，保持原共同 globaltimer span × 本 trial unit 全 warp
clock/timer 比率中位数策略，不相减不同 SM clock64。保留所有raw，连续3/4
分组再聚合，记录每 trial ratio、SM、实际 launch 和原单位完整 warp 覆盖。
最终独立同码及单位验收仍归 ledger owner；输入合同本身不自动匹配。

6例`prefill_clean` attention每trial连续enqueue两个kernel（`sm90_fp8_mqa_logits`
bank0＋`smxx_clean_logits` bank1），bank只是host侧marker指针偏移，launch记录与
SASS均看不到。补证采用物理事实而非符号或顺序猜测：full7 host源已有的
`get_sm90_suite_attention_marker_ptr`在计算传给即将launch的kernel的bank指针处，
经既有`dlsym(RTLD_DEFAULT)`接点把`(bank, marker_ptr, logits_ptr, payload_bytes,
marker_bytes)`写入同一loaded_code通道（observer新增`fi_deepgemm_attention_bank`，
事件携带launch_cursor与host thread）。消费者按cursor+1连接紧随的物理launch，
要求pid/thread一致、同cursor唯一、窗口内单context、每launch恰一条bank事件，
并核对`marker_ptr == logits_ptr + payload_bytes + bank*marker_bytes`与不同bank按
marker_bytes间距；unit仍按原kernel_name+同selected Function指纹选中launch，覆盖、
去重与cluster核对仅取该launch的bank行，全部两bank raw照旧导出。缺事件、歧义、
算术不符、无bank的helper launch或跨线程/进程/context一律fail-closed拒收整trial；
历史whole-operation事实不变。
attention每次launch()内部分配输出时，full7分配helper以ATen fill（CUDA runtime API）
清零marker bank区，该launch落在trial窗口内且无driver身份：observer新增
`fi_deepgemm_attention_reset`，host patch在`.zero_()`前上报被清零字节区间（同dlsym
接点与cursor/thread纪律）。core的layout transpose同理：`get_mn_major_tma_aligned_tensor`
对输出storage的marker尾区`narrow(...).zero_()`（grid 3584实测量级），同一reset接点上报；
其非连续输入的`out.copy_(batched_sf)`回退路径保持无attest，按下述foreign槽冲突规则处理，不伪造reset。
消费者把reset事实（cursor+1、pid/thread一致）接上的runtime launch当作
marker清零，不把它当选中目标。有bank事件时还要求区间按marker_bytes对齐且覆盖全部claimed bank指针窗口。
无bank事实时，观测槽必须唯一对应一条物理driver enqueue，选中Function必须就是这条enqueue；
同一Function的另一grid留在记录里，不并进该unit的槽。未接上reset的runtime launch保留为
foreign：名字撞上选中Function、参与槽等于选中unit观测槽、或geometry无法比较，都拒收该unit；
其余foreign不阻断已经证明的选中unit。窗口context检查仅计driver launches。
warp普查按marker仪控自身的容量：`sm90_suite_marker`的begin/end对`warp>=kMaxWarpsPerCTA`
直接return。容量只来自该次package `sm90_suite_marker.json`的`max_warps_per_cta`
（通用marker格式常量，不按case写死）。block 640（20 warp）在容量16时只有warps 0-15
能有marker行——硬件与仿真同二进制同仪控，普查范围=`min(block//32, capacity)`，
缺一行仍拒收。`unit_scope`记录`marker_warp_capacity`、`block_warps`、`census_scope`，
并从该trial实际unit bank行导出`clock_marker_warps`（各CTA的warp集合必须一致，
否则拒收；不能把20个block warp说成都有marker）。`member_warp_binding`带同一列表。
不再写`all_unit_warps_required`：该布尔在容量截断后会把“参与集合齐全”说成
“block内全部warp都有marker”。campaign比较须改读`clock_marker_warps`，
与捕获侧同名列表对齐同一outermost-pair参与集合；本文件不改campaign。

## 验证边界

`harness/tests/test_actual_binding.py` 仅测试消费者 boundary/errors：缺warp、
重复行、不完整七字段、clock非正、geometry/code/policy不符、enqueue失败、
context不符、unload、多次enqueue、日志截尾/截名、重复Function，以及bank绑定
（多bank正例仅归约unit bank、缺事件/歧义cursor/指针算术错/双launch同bank/
未claimed bank行/跨线程或跨进程/跨context均拒收）、reset绑定（core形reset-attested
fill正例、同槽重复enqueue在保留记录后拒收、不同grid的同一Function不抢走唯一槽主、
两条选中目标enqueue拒收、foreign槽或名字撞车拒收、槽不同的foreign不阻断已绑定bank、
宽block普查=记录的producer容量内warps且导出
`clock_marker_warps`、缺一行拒收）。`test_bare_campaign.py`拒绝缺layout reset的host包。
测试fixtures不是 GPU 证据；H100真实代表与真实raw副本负控的 rc/machine/command/artifact
由独立外置运行原件证明。没有运行的路径不能宣称已验证。

`check_bare_evidence.py --run NEW_EXTERNAL_RUN --unit-json ORIGINAL_CAPTURE_UNIT_JSON
--out NEW_CONTROLS --observer EXTERNAL_LIBACTUAL_OBSERVER_SO`在获准job中消费
真实bare原件，独立保存缺warp/重复warp/缺字段/clock非正/截尾/截名/重复Function
副本，并对真实多bank run保存真删bank事件/歧义cursor/指针算术错副本（单bank run
如实记not_applicable），绝不修改原raw/result。它还以代表实际cubin走旧ModuleLoad/GetFunction，
调用原driver真实失败的零grid/block enqueue（不执行kernel），检查消费者拒收，
保存真实return code及module/unload原件；并在隔离cache、无preload的子进程
检查强制observer缺失拒收。副本负控不是新GPU测量，不可作为新分母或actual成功。
删除reset事件的副本若仍由bank/唯一槽事实证明目标enqueue，且对应runtime launch
在消费结果中原样保留为不冲突的foreign，则记accepted_foreign_launch_retained，
不是拒收负例；bank歧义、指针算术错误和目标槽/名字冲突仍拒收。observer缺失
子进程使用当前harness的PYTHONPATH和cwd，必须命中实际host接点断言，模块导入失败不算。

`assemble_bare_handoff.py`接收原100 scope、原unit-root及所有真实case-root，
严格每case唯一地索引receipt/result/raw/source/load/launch/SASS及其SHA，保存
首trial实际module/Function/完整config及全部trial原件指针。它不写463 ledger，
不授予独立同码精度eligibility；14例attention中8例单kernel单bank按通用规则直接
产出unit分母，6例prefill_clean经bank绑定事实产出，仍无闭合证据时保留明确
unmeasured原因，不以whole-operation替代。
例：在获准job内用`--case-root REPRESENTATIVES/cases --case-root REMAINING/cases`
拼接代表与补集，不重测已成功case；缺少或重复原件是错误，不缩scope为已成功集合。
