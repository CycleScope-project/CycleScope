# Hopper 与 Blackwell：差在哪

[English](../../en/design/arch-semantics.md)

一条管线服务两个架构。架构差异只在四处进入：tracer 的架构表、转换规则、机制树、引擎的架构模块。其余部分——记录流、段格式、仿真核心——都是共享的，`cpp/core/` 里没有设备分支。

## 1. 差异在哪一层

| 层 | 共享部分 | 架构部分 |
|---|---|---|
| 采集 | `tools/tracer/tracer.cu`、`record_format.h` | 架构表（`arch_table.h`、`arch_hopper.h`、`arch_blackwell.h`）：操作数族、mbarrier 字规则、TMA 坐标、描述符记号 |
| 转换 | `cpp/{capture,convert}/`、`python/cyclescope/convert/` | `cpp/arch/<arch>/{capture_rules,convert_rules}.cpp`、`python/cyclescope/arch/<arch>/` |
| 机制 | `configs/mechanism_tree/common/` | `configs/mechanism_tree/{hopper,blackwell}/`，生成为 `configs/trees/{h100,b300}.json` |
| 引擎 | `cpp/core/` | `architecture_module.hpp` 后面的 `cpp/arch/{hopper,blackwell}/` |

仿真时由 launch 的 `LNCH.device` 选模块（`cpp/arch/registry.cpp`）和作业构建时用的树。

## 2. 共享语义

两个架构只处理一次的语义：mbarrier 的 init/arrive/expect/wait 相位、CTA 命名 barrier 与 cluster barrier、TMA 与 bulk 搬运、`cp.async` 组、通用访存、借全局存储做的同步。它们的 `SEMA` kind 与引擎状态机都与架构无关（[engine.md](engine.md) §5）。

## 3. Hopper：warpgroup 与 WGMMA

Hopper 的张量核由 **warpgroup**——四个 warp 组成的一个集合体——驱动，结果返回到寄存器。转换把指令映射到语义 kind（`python/cyclescope/arch/hopper/semantics.py`）：

| 指令 | 语义 kind |
|---|---|
| `WARPGROUP.ARRIVE` | `warpgroup_arrive` |
| `HGMMA` / `QGMMA` / `IGMMA` / `WGMMA` | `wgmma`（形状、数据类型、ss/rs 源模式、累加寄存器） |
| `WARPGROUP.DEPBAR.LE`、`WARPGROUP.WAIT` | `warpgroup_wait` |
| `FENCE.VIEW.ASYNC.S` | `descriptor_visibility`（代理 fence） |

在引擎里（`cpp/arch/hopper/`），模块运行集合发射会合（四个 warp 都到达后操作才上共享张量流水线）、`AsyncGroups` warpgroup 通道上的组 FIFO、向消费结果的指令的累加器交接，以及操作在途时的寄存器堆发射窃用。`proxy_fence` 建模 fence 等待的异步代理排空。

## 4. Blackwell：UMMA、张量存储、CLC

Blackwell 的张量核由单线程发起，结果累加在**张量存储**（TMEM）——一个 CTA 按预算分配的每 SM 列池里。转换映射（`python/cyclescope/arch/blackwell/semantics.py`）：

| 指令 | 语义 kind |
|---|---|
| `UTCHMMA` / `UTCQMMA` / `UTCOMMA` | `umma_issue`（kind、cta_group、A 源、形状、描述符） |
| `UTCBAR` | `umma_commit`（对象、cta_group、掩码、范围） |
| `UTCATOMSWS`（`FIND_AND_SET` / `AND` / `RELINQUISH` 形式） | `tmem_alloc` / `tmem_dealloc` / `tmem_relinquish` |
| `LDTM` / `STTM` / `UTCCP` | `tmem_load` / `tmem_store` / `tmem_copy` |
| `FENCE.VIEW.ASYNC.T` | `tmem_wait_store` |
| `UGETNEXTWORKID` | `clc_query` |
| `STAS` | `shared_store_async` |

在引擎里（`cpp/arch/blackwell/`），模块运行 UMMA 流水线节拍与 commit 到达（`cta_group::2` 的操作投在对首 CTA 上）、TMEM 列预算——预算耗尽时 alloc 阻塞、dealloc 释放并唤醒——TMEM load/store/copy 的定时，以及 CLC 查询经其 mbarrier 完成。
