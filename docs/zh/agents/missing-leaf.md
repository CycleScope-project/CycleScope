# 缺叶 / 缺参数

[English](../../en/agents/missing-leaf.md)

机制存在，但叶参数缺失、是占位值，或离散轴值无对应叶行
（`unsupported:engine:discrete_miss`）。

## 1. 找到叶

1. `cyclescope tree show <mechanism>` 从
   `configs/trees/{h100,b300}.json` 打印该机制的节点与叶。
2. 源片段是每机制一个文件
   `configs/mechanism_tree/{common,hopper,blackwell}/mechanisms/
   <mechanism>.json` 与 `…/leaves/<mechanism>.json`，另有各目录
   下的 `axes.json`、`device.json`、`resources.json` 与出
   处文件 `sources.json`。
3. 证据等级：`measured` / `experimental` / `structural` /
   `placeholder`。当前 `sources.json` 计数 —— hopper：95
   measured、57 experimental、10 structural、205 placeholder；
   blackwell：195、78、21、4；common：4、4、4、3。

## 2. `discrete_miss` 症状

离散叶是轴值表（如访存宽度或 warp 数）。
`unsupported:engine:discrete_miss`
（`cpp/core/mechanism_runtime.cpp`）表示仿真值落在已记录行之
间/之外 —— 先看片段中 `discrete_axis` 约定项确认已记录域，再
补测缺的格子（下述流程）或确认轴值本身是事实问题（转采集/转
换手册）。

## 3. 测量流程（按仓库实践）

1. 先写**并冻结** `tools/microbench/<name>/protocol.json`。真
   实协议使用的键：`schema`
   （`cyclescope.microbench-protocol.v1`）、`name`、`version`、
   `frozen_before_gpu`、`question`、`mechanisms`、`ownership`、
   `devices`、`probe`、`cells`、`sass_gate`、
   `predictions_before_data` /
   `prediction_of_the_installed_model…` / `alternative…`、
   `acceptance`（安装判据）、`decision`、`install`、
   `controls`、`runs`、`not_allowed`、`addenda`（带日期修正）、
   `supersedes`。
2. probe + SASS 门：probe 须编译出预期 SASS；门记录在
   `sass_gate` 下，由协议自带脚本检查。
3. 每设备两跑，环境变量驱动：

   ```
   PROTOCOL=<name> DEVICE=h100|b300 RUN=<tag> \
       [CELL=<subdir>] [OUT_ROOT=<dir>] [NCU=<ncu>] [TIMEOUT=3600] \
       bash tools/microbench/run_local.sh
   ```

   它在 `tools/microbench/<name>` 下跑驱动的 `run.sh` /
   `run_probes.sh` / `run_ncu.sh`，输出到
   `<artifact_root>/microbench/<dir>/<device>-<run>`。两跑都保
   留，不挑数据。
4. 模型脚本 + `acceptance` 判据决定**装或 unrealised**
   （`decision`/`install` 字段）；判据不过则不装 —— 结果记入
   协议。
5. 修正以带日期 `addenda` 写入，绝不改原预测。
6. 把值装进叶片段 + 带证据等级的 `sources.json` 记录。
7. `cyclescope tree build && cyclescope tree validate --strict`
   —— 五类约定（`field_model`、`discrete_axis`、`unit`、
   `evidence_crossing`、`class_resource`）全 0；若改了
   `configs/schema_registry.json`，用 `cyclescope tree build
   --emit-cpp` 重新生成
   `cpp/core/generated/parameter_schema.hpp` 与
   `python/cyclescope/tree/generated_schema.py`。
8. 重跑出货示例与受影响 case；反例如实报告。

绝不因为某 case 误差变小而选值 —— 先归因，再测量。
