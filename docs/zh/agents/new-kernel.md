# 采集新 kernel

[English](../../en/agents/new-kernel.md)

为能在本机运行的应用采集并仿真一个不在套件中的 kernel。

## 1. 前置条件

1. 已供应的 tracer：`$CYCLESCOPE_ARTIFACT_ROOT/build/tracer/
   sm_90a|sm_103a/tracer.so`（`provision_tools.py build
   tracer-sm_90a`，见 [../capture-tools.md](../capture-tools.md)）。
2. 站点配置能解析引擎构建（`configs/site.json`，由
   `configs/site.example.json` 复制，或 `CYCLESCOPE_BUILD_DIR`）与
   `CYCLESCOPE_ARTIFACT_ROOT`。
3. 应用在其正常环境下可独立运行。

## 2. Hopper case 如何定义

Hopper case 来自 `suites/hopper/<set>/cases.tsv`（每行一个，制表
符分隔；解析在 `python/cyclescope/capture/driver.py`）：

| 列 | 含义 |
|---|---|
| `set` | 库/集合标签（信息性分组） |
| `case_id` | 唯一 case 名 |
| `kernel_regex` | 对 mangled kernel 名的正则；选择被跟踪的 launch |
| `occurrence` | `kernel_regex@occurrence` 中名字匹配正则的第 N 个 launch（1 基，全程计数；tracer 对 demangled 与 mangled 名都试匹配） |
| `argv_json` | JSON 列表形式的 argv，`"` 要双写为 `""`；路径可用下文占位符 |
| `cwd` | 工作目录，`-` 为默认 |
| `env_json` | 附加环境 JSON 对象，`-`/`{}` 为无 |
| `env_profile` | 空/`default` 或 `far`（把 argv 包进 `bash -c 'source suites/hopper/runners/environment_far.sh && exec "$@"'`） |
| `tma_recorder`、`baseline` | 出货套件中为 `-` |
| `note` | 自由文本（分号分隔的元数据） |

`argv_json`/`env_json` 内路径形式：`$CYCLESCOPE_ARTIFACT_ROOT/
tools/<name>` 指供应工具，`$CYCLESCOPE_ROOT` 指仓库内文件，
`<CAPTURE>` 指 per-case 采集目录（运行时生成的输入放这里）。

launch 选择优先级（`python/cyclescope/capture/driver.py` 的
`_capture_selection`）：`--reference` 的 case 键 >
`launches.tsv` 表行（`resolved.launch_key`，精确
`(kernel, occurrence)` 对）> ncu census。只有 census 这一路会跑
无指标 ncu prepass（`ncu_launches.jsonl`；需要 `ncu`，见
`CYCLESCOPE_NCU`）—— 参照键或表行会跳过它。Hopper 上 census
只是交叉检查 `occurrence` 个匹配存在（`occurrence N of <regex>
beyond ncu census`）；census 缺失被记录而非致命。

## 3. Blackwell case 如何定义

`suites/blackwell/<set>/cases.txt` 把 `case_id` 映射到 family；
family 行在 `suites/blackwell/<set>/cases_<family>.json`，family
适配器在 `suites/blackwell/runners/common/capture_adapters.py`
（`capture_registry.json` 按 case-id 前缀路由到适配器）。新
family 需要适配器代码；已有 family 的 argv 级 case 只需一行
JSON。未在发布树上实操验证。

## 4. Hopper 实操示例

1. 建 `suites/hopper/local/cases.tsv`，表头复制
   `suites/hopper/full_gpu/cases.tsv`，加一行：

   ```
   local	my_demo	my_kernel_regex	1	[""/path/to/app"",""--flag""]	-	{}	default	-	-	standalone binary
   ```

2. 干跑解析（不需要 GPU）：

   ```
   python3 -m cyclescope.cli capture --suite hopper/local --case my_demo --out <scratch> --dry-run
   ```

   输出一段 JSON。上例在 scratch 副本中的真实输出（以
   `/bin/true` 为应用）：

   ```json
   {
     "suite": "hopper/local", "case": "my_demo",
     "argv": ["/bin/true"],
     "env": {"PYTHONPATH": ".../suites/hopper/runners/harness:..."},
     "cwd": "",
     "tracer": "<artifact root>/build/tracer/sm_90a/tracer.so",
     "mode": "both",
     "capture_format": "cyclescope.capture.v1",
     "launch_selection": "my_kernel_regex@1"
   }
   ```

   （`tracer` 为缩写 —— 真值是已供应 tracer 路径。）核对
   `argv`/`env`/`cwd` 与应用需要一致（driver 会把 hopper
   harness 目录注入 `PYTHONPATH`）。
3. 在 GPU 机器上运行：
   `cyclescope run --suite hopper/local --case my_demo --out <out>`。
4. 判读：没有 baseline 行，`cyclescope report` 判 `n/a`；读
   report 行的 `ratio`/`err_pct` 列（`sim_cycles`/`hw_cycles`），
   与相邻出货 case 的 ratio 对照。

## 5. 目前不支持

- 没有"跟踪任意命令行"的入口；case 一定经套件行解析（Hopper
  的 `cases.tsv`，Blackwell 的 family 规格）。
- 本例的 GPU 阶段（真实卡上的采集 + ncu 参照）未在发布树上实
  操；干跑解析已实操。
- Blackwell 新 family 需要适配器代码；Hopper 独立二进制只需一
  行 `cases.tsv`。
