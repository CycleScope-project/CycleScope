# cells-v2 分桶统计

| 类别 | 数量 (两设备相同) |
|---|---|
| probe:alu | 49 |
| memory | 29 |
| synchronization_and_async_control | 22 |
| probe:alu3 | 14 |
| control | 16 |
| probe:misc | 16 |
| probe:alu2 | 15 |
| tensor | 13 |
| probe:svc | 10 |

## alu3 需新增格（未覆盖的寄存器类）

- DSETP：lat.DSETP 依赖链 + thr（按 alu.cu 方案）
- F2F：lat.F2F 依赖链 + thr（按 alu.cu 方案）
- FSETP：lat.FSETP 依赖链 + thr（按 alu.cu 方案）
- HADD2.F32：lat.HADD2.F32 依赖链 + thr（按 alu.cu 方案）
- IMAD.HI：lat.IMAD.HI 依赖链 + thr（按 alu.cu 方案）
- LDTM：lat.LDTM 依赖链 + thr（按 alu.cu 方案）
- LEA：lat.LEA 依赖链 + thr（按 alu.cu 方案）
- MOV：lat.MOV 依赖链 + thr（按 alu.cu 方案）
- PLOP3：lat.PLOP3 依赖链 + thr（按 alu.cu 方案）
- STTM：lat.STTM 依赖链 + thr（按 alu.cu 方案）
- UFLO：lat.UFLO 依赖链 + thr（按 alu.cu 方案）
- VHMNMX：lat.VHMNMX 依赖链 + thr（按 alu.cu 方案）
- VIADD：lat.VIADD 依赖链 + thr（按 alu.cu 方案）
- VIADDMNMX：lat.VIADDMNMX 依赖链 + thr（按 alu.cu 方案）

LDC/ULDC/LDCU 注释：2026-11-05 首次有效运行前改归 memory 桶——两设备的 memory.constant_cache 已装 crossings v2 协议值（f2a000f），按 ownership_rule 原文适用 1/1/0；alu3 的 ldc_chain 格删除（计数表中 probe:alu3 17 → 14，memory 26 → 29）。
