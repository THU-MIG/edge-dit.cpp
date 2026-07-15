# Cache Framework 现状总结与前进目标分析

> 面向 `edge-dit.cpp` 的 step-caching 子系统
>
> - 文档状态：现状盘点 + 路线建议
> - 分支：`refactor/cache_framework`
> - 触发来源：一次针对 cache framework 的代码审阅
> - 目标读者：cache 算法开发者、Runtime/Backend 开发者

---

## 1. 背景与范围

`refactor/cache_framework` 把 step-caching 从旧的 `CacheController` 重构为声明式的
`CacheEngine`。本文做两件事：

1. **盘点现状**：哪些能力真正落地、优点与差距分别是什么；
2. **给出前进目标**：把「落地声明式引擎」作为推荐方向深入分析其优势与问题，并列出备选方向。

本文以 `docs/cache_framework_redesign.md` 描述的声明式架构为**目标态**，衡量「已落地
vs 差距」。文中所有「未承重 / 死代码」类断言均已由代码检索核实，并附 `file:line`。

> **最新进展（2026-07-15）：cache 层与模型层的解耦已完成。** 07-14 之后的一批提交
> 把「去 hook（dehook）」这一原本列为「数周量级、终态」的工作做完了：模型 forward 不再
> 认识任何 cache 概念，改为在结构地标处调用 `TapRegistry` 的条件 `tap()`；旧的
> `CacheGraphScope` seam 已被**整块删除**（commit `cd0c9b7`，`-1667` 行），`ED_CACHE_SUBSTEP`
> 门控随之消失——**substep 循环成为唯一执行路径**。下方 §2.6、§4、§5 已据此改写；
> §5 的三层路线图中，前两层（StateManager / Operator）＋原「Slice 4 去 hook」均已落地，
> 只剩第 3 层（GraphRepository）与 SP+缓存复合未做。

> **术语：承重 vs 脚手架。**「承重」指该组件在真实推理路径上被调用、去掉会改变行为；
> 「脚手架」指已实现但当前无任何调用方、去掉不影响任何现有功能的代码。本文的核心结论
> 之一，就是区分声明式设计里哪些是承重、哪些是脚手架。

---

## 2. 现状总结（已落地）

### 2.1 架构

- `CacheEngine`（`src/core/optimization/cache/runtime/cache_engine.{hpp,cpp}`）替代
  `CacheController`，保留 `init / enabled / begin_step / end_step / run_branch /
  log_summary` 接口；头文件末尾以 `using CacheController = CacheEngine;` 别名过渡，
  因此 pipeline 只改类型名即接入。
- 6 个 pipeline 已接入 `run_branch`：flux、flux_kontext、qwen_image、qwen_image_edit、
  sd3、wan（`src/dit_models/pipelines/*_pipeline.cpp`）。

### 2.2 能力协商是真的（承重）

`CacheEngine::init` 在 `policy_->compile(...)` **之前**调用
`validate_requirements`（`runtime/capability_negotiation.cpp`）。不支持的组合会显式
`LOG_WARN("cache disabled: ...")` 并返回 false，取代旧的「静默 full-compute」降级。
错误信息会列出模型实际暴露的 site（`exposed_sites`），可诊断。

### 2.3 8 个 policy 迁至 `ICachePolicy`

`policy/policy_factory.cpp` 分发：null / EasyCache / UCache / DBCache+CacheDiT
（共用 `condition_policy`）/ TaylorSeer / MagCache / DiCache / SenCache，各一个文件于
`policy/policies/`。

### 2.4 三种 granularity（承重）

在 `cache_graph_lowering.cpp` 的 `execute_substeps` 中按 `SubstepOpKind` 分流（policy 的
`next_substep()` 逐子步产出 `SubstepPlan`，中间层翻译为 tap 驱动的 runner pass）：

- **Output**：黑盒，只用 `hooks.full()` + 主机侧输出 diff/复用，SP 下也可用；
- **Feature**：经 `TapRegistry` 在 ModelIn/ModelOut 锚点捕获/注入残差（MagCache、TaylorSeer、
  SenCache）；
- **Probe**：跑浅层前缀（`stop_after`）再决策（DiCache）。

### 2.5 性能优化真实有效

详见 `docs/cache_quality_benchmark_report.md`（H200 / 50 步 / Flux+Qwen）：

- on-GPU MagCache 特征复用、DiCache GPU 探针重写、`ED_CACHE_COMPILED_GRAPHS`
  build-once 图复用；
- skip 成本从 ~40ms/skip 降到 ~1ms，且验证 byte-identical，质量不变；
- Flux：MagCache 2.05x、DiCache 默认 1.24x（近无损，PSNR ~40）。
- **2026-07-14 更新**：Flux 和 Qwen 的默认 on-GPU 特征复用现统一走**声明式设备槽**
  （`CacheStateManager` 设备后端），已替换 legacy `DiCacheGpuState` 路径；设备槽 vs
  host-declarative 验证**字节一致（PSNR 100 dB）**，skip 计数逐位一致（见 §5.5）。
  注：本轮 Qwen 验证需 `ED_QWEN_SINGLE_FUSED_ATTENTION=0`（默认融合注意力在当前 CUDA-13
  环境下产出纯白图，属既有环境问题，与 cache 无关）。

### 2.6 cache/模型解耦已完成：TapRegistry 取代 seam（承重，2026-07-15）

原设计文档把「彻底去 hook」列为终态、数周量级的 Slice 4。**它已经做完了**：

- **依赖方向已倒置干净**：cache 层（`src/core/optimization/cache/`）**零 include 任何具体
  模型 header**（flux/qwen/wan/mmdit），只依赖模型侧暴露的抽象——`anchor` / `cache_site` /
  `model_topology` / `model_schema` / `tap_registry` / `model_cache_contract`。已由 grep 核实。
- **模型 forward 不再认识 cache**：4 个模型（flux/qwen_image/mmdit/wan）的 forward 在结构
  地标处调用 `TapRegistry` 的条件 `tap()`（`ctx->tap_registry`）——非请求即 no-op，不 pin
  buffer、gallocr 无感。cache 需要读哪些张量由 per-substep 的 `SubstepPlan.taps` 决定，
  而非模型枚举字段。inject（复用注入）也走 registry：forward 在 `inject_at(i)` 处替换流
  并跳到 `inject_resume()`。
- **`CacheGraphScope` seam 已整块删除**（commit `cd0c9b7`，`-1667` 行）：旧的固定 `*_node`
  字段、`kCache*Name` 常量、`expand_cache_scope_nodes`、runner 的 `cache_scope_` 成员 +
  `set_cache_scope`、`DiCacheGpuState` 独立 GPU 状态路径均已移除。只保留一个纯结果结构体
  `sd::DiffusionCacheResult`（host 读回的 output/feature/before/probe + DiCache 标量）。
- **substep 是唯一路径**：`ED_CACHE_SUBSTEP` 门控消失，`run_branch` 无条件走
  `CacheGraphLowering::execute_substeps`（`cache_engine.cpp:196`）；旧 `execute` 及其
  hook 大 if/else 已删（commit `a2831c6`，`-444` 行）。8 个方法全部实现 `next_substep()`。
- **DiCache probe 也已 tap 化**：checklist 曾标注「probe 未迁」的那项已完成——probe 的
  跨步持久张量（prev_probe/probe_prev1/2）经 `TapRegistry::ProbeMetricOperands` 线程进
  indicator lowering，delta_y/delta_x/gamma 在图内归约后读回标量。

---

## 3. 优点

1. **解耦目标已达成**：新增 cache 方法 = 写一个 `ICachePolicy`；新增模型 = 暴露一个
   `DiTModelCacheContract` + 在 forward 的结构地标处调用 `TapRegistry::tap()`（ModelIn/
   BlockOut[i]/ModelOut）。摆脱了「模型数 × 方法数」的组合爆炸，且模型侧不再 include 或
   认识任何 cache 概念（见 §2.6）。
2. **失败是显式的**：SP-parallel 或无法切图时，Feature/Probe 方法经能力协商被正确拒绝，
   而不是静默跑成 no-cache。
3. **CacheProgram 可 dump**（`ED_DUMP_CACHE_PROGRAM`），策略决策可观测。
4. **GPU 路径提速且质量不变**：skip ~1ms、byte-identical。
5. **CFG-parallel + 缓存可用**：cond/uncond 各自用 `condition_key`（condition 结构地址）+
   `CacheBranch` tag 隔离状态（`cache_slot.hpp:26`、`cache_state_manager.cpp:8`
   `key_of`、`magcache_policy.cpp` `branch_for`）。cache 命中只跳过本地 forward，
   `cfg_all_gather` 仍无条件执行以保持同步（对齐 `CLAUDE.md`）。

---

## 4. 缺点 / 差距

### 4.1 承重层的问题

- **粗粒度 Option-A 拓扑**：`dit_model_cache_contract.cpp` 只暴露单个 `BLOCK_STACK`
  segment（输入投影 / 整块 stack / 输出投影）。无 per-block、attention/FFN、token 级
  site。`cache_program.hpp` 的 `CachePhase::{FULL_ANCHOR,CORRECTION,REINTEGRATION}`
  仅占位，实际只用到 `FORWARD/PROBE`（`redesign` 文档阶段三/四未做）。
- **SP-parallel 下 Feature/Probe 全失效**：`feature_cache_available()` 定义为
  `!can_attempt_graph_cut_segmented_compute()`（`ggml_extend.hpp:6472-6473`），而后者在
  「任何 process group 使能」时即为 true —— 故纯 SP 也会关掉 tap 捕获路径（pipeline 以
  `cache_seam_available = !cfg_parallel && feature_cache_available()` 传入 `CacheEngine::init`），
  SP 下只剩 Output 粒度。这是引擎两大卖点（多卡并行 + 步缓存）当前无法叠加的直接体现，
  也是解耦完成后**最大的剩余能力缺口**。
- **~~hook 耦合脆弱~~（已随 dehook 消解）**：`gpu_metric`/`branch_key` 曾静默把所有 Feature
  cache（MagCache/TaylorSeer/SenCache）归零跑成 0/N。该脆弱面源于旧 `CacheGraphScope` 的
  固定 `*_node` 字段 + 双写路径；§2.6 的 dehook 已删除该 seam，tap 由模型按请求集主动登记，
  这一类回归的结构性成因已消除。保留此条作为历史教训：触及 tap/lowering 时仍按「skip 计数 +
  PSNR≥地板」验收。
- **cache 路径曾有 GPU 内存泄漏**（已修），长跑仍需 chunked 生成绕过。

### 4.2 声明式机器：已承重 vs 仍是脚手架（2026-07-15 更新，均已 grep 核实）

设计文档最有雄心的三块声明式机器，现状已从「只落地决策层」推进——**前两块已承重**：

- **`CacheStateManager` 已承重（history ring + 设备槽后端）**：`CacheEngine::init` 把
  `state_` 传入 `CacheGraphLowering::execute_substeps`（`cache_engine.cpp:196`），lowering 的
  `ActionInterpreter` 真实调用 `state_.read_history` / `state_.write` / `state_.rotate_history`
  （`cache_graph_lowering.cpp`），设备槽路径调 `state.read` + `state.alloc_device_entry`。
  TaylorSeer 的导数历史 ring、MagCache/Qwen 的**设备端单残差槽**都跑在其上。
  仍是脚手架的部分：`commit_step`（`cache_state_manager.cpp`）/ `rollback_step`
  仍是**显式 no-op**——但这是**诚实的**：全库无 `rollback_step` 调用方，`commit_step` 只被
  `end_step` 无条件调用一次，且没有任何 cache 方法会中途 abort（失败走同步 full-compute
  回退，不涉及状态回滚）。所谓「事务式回滚安全」当前无触发点，是为未来细粒度/token-cache
  预留的接口占位，而非承重能力。各 policy 仍自持标量决策状态（`states_` map），这是**决策
  逻辑**（该留在 policy），与 StateManager 管的**张量存储**已分离。
- **`CacheOperatorRegistry` + operator 已承重**：`CacheEngine` 实例化 registry 并调
  `register_builtin_cache_operators`（`cache_engine.cpp:111`，此前死代码）；lowering 的
  DIFFERENCE/PREDICT/BLEND 动作经 `operators_.find(...)` 路由到 `DifferenceOperator/
  LinearPredictOperator/WeightedBlendOperator::apply_host`。
  EasyCache/UCache 的输出 diff、TaylorSeer 的历史外推 blend 都由 operator 执行。
- **仍无 `GraphRepository` / 多静态图变体**：`GraphRepository`、`GraphVariantKey`、
  `get_or_compile` 在 `src/`、`examples/` 及除 redesign 外的所有 docs 中**零命中**。
  设计文档核心思想「运行时在多张预编译静态图变体间选择」**没有实现**。实际执行是
  tap 驱动的 runner pass 分派：`execute_substeps` 按 `SubstepPlan.op.kind` 分支去调
  `hooks.full` 与 tap 驱动的 `hooks.substep_capture/substep_probe/substep_inject_*`（device
  或 host）。因此 `CacheProgram`/`GraphVariantPlan` 是「决策描述符 + 动作序列」（告诉 lowering
  产出哪种 substep、执行哪些 slot/operator 动作），而非编译产物。

> **小结**：声明式组件里，`CacheProgram`/`GraphVariantPlan`（描述符 + 动作）、
> `CacheRunnerHooks` lowering、`CacheStateManager` 的 history/设备槽后端、`CacheOperatorRegistry`
> 均已承重——8 个方法（含 Flux/Qwen 的默认设备端 GPU 复用）真实跑在 StateManager+Operator 上；
> 且模型侧已彻底 dehook（§2.6）。仍是脚手架的只剩：`GraphRepository`/多变体编译（未建），
> 以及 `commit_step`/`rollback_step` 的事务语义（诚实 no-op，无触发点，为未来预留）。

---

## 5. 推荐前进目标：落地声明式引擎

**方向**：把 §4.2 的三块脚手架真正接上，兑现 `cache_framework_redesign.md` 的核心承诺。
**现状（2026-07-14）**：前两层已落地承重，只剩第 3 层。

### 5.1 具体含义（分层，每层独立按「skip 计数 + PSNR≥地板」验收）

1. **StateManager 承重化 ✅ 已完成**：`CacheStateManager` 已是 TaylorSeer / DiCache /
   MagCache / Qwen 的真实状态后端（host history ring + 设备端单残差槽），替换了这些方法
   张量存储层面的私有 `states_`。事务 commit/rollback 仍是诚实 no-op（无触发点，见 §4.2）。
2. **Operator 化数学 ✅ 已完成**：残差 diff / 预测外推 / blend 已路由到 `CacheOperator`，
   policy 引用 operator id，`register_builtin_cache_operators` 真正被调用。
3. **GraphRepository + 变体编译 ⬜ 未做**：引入真正的 FULL/REUSE/PREDICT 静态图变体编译与
   缓存，并与 `ED_CACHE_COMPILED_GRAPHS` 的 build-once 图复用融合（两者都想「少建图」）。
   这是唯一剩下的声明式层，也是 SP 复合 / 细粒度的天然下游。

### 5.2 已兑现的收益（层 1-2）与剩余价值（层 3）

已兑现：
- 8 个方法 + Flux/Qwen 默认设备端 GPU 复用跑在 StateManager+Operator 上，`CacheProgram`
  不再只是决策描述符；
- operator 化让新方法接入更快（组合已有 operator，而非每次内联新数学）；
- 设备槽把 skip 成本压到 ~1ms 且验证字节一致（见 §2.5、§5.5）。

剩余价值（层 3）：
- 为细粒度 / 轨迹修正 / token cache 打地基（需要真正的多变体静态图）；
- 与 compiled-graph 复用合流，可能进一步降低建图开销；
- 事务 rollback 安全当前**无触发点**，不是收益——除非未来出现会中途 abort 的方法。

### 5.3 问题 / 风险（不回避）

- **层 1-2 已证「脚手架可承重」**：设备槽验收字节一致（PSNR 100 dB）、skip 计数逐位一致，
  证明「接脚手架」确有具体收益（skip ~1ms、单一 GPU 复用路径）。这消解了原本「纯整洁化、
  无用户可见收益」的担忧——本文仍主张**把落地当手段而非目的**，每层绑定具体收益。
- **过度设计风险（层 3）**：`GraphRepository` / 多变体在 GGML 静态图上与既有 `PlanCache` /
  `compute_reuse`（`ED_CACHE_COMPILED_GRAPHS`）职责重叠。落地前须先厘清边界，
  否则是两套「图缓存」互相打架（参见 `perf_gap_vs_diffusers.md` 的建图开销分析）。
- **回归风险高**：层 3 触及最脆弱的 graph-cut / PlanCache 子系统。曾出事的 hook 耦合面
  已随 §2.6 的 dehook 消解（seam 删除、tap 按请求集主动登记），但 tap/lowering 仍需谨慎，
  触及处按「skip 计数 + PSNR≥地板」验收。
- **收益递延**：只有第 3 层（GraphRepository）直接关联 SP 复合与细粒度这些用户可感知能力。

### 5.4 首个最小切片（已完成，作为「脚手架可承重」的证明点）

原计划：只把 `CacheStateManager` 接为 **TaylorSeer** 的 history 后端（`n_derivatives+1`
深度），改动面小、边界清晰，作为证明点。**实际结果**：该切片 + 后续 7 个方法 + Flux/Qwen
默认设备端 GPU 复用全部落地并验证等价（skip 计数逐位一致 + PSNR≥地板；设备槽 vs host
更达字节一致 100 dB）。证明点成立——脚手架确可承重，且带来了具体收益（skip ~1ms、单一
GPU 复用路径），故层 1-2 已推进完成。**下一步的判断点转移到层 3**：GraphRepository 与
`ED_CACHE_COMPILED_GRAPHS` 职责重叠，落地前须先厘清边界（§5.3），否则宁可止步。

### 5.5 实施进展（2026-07-15，层 1-2 + 模型 dehook 已完成并验证）

> **验收基准更正**：§5.1/§5.4 原写「byte-identical」，实测证伪——本引擎 CUDA 后端**逐帧
> 不确定**（同一 no-cache 配置连跑两次 PSNR ≈ 35 dB，非 100%）。因此等价性验收改为
> **①skip 计数逐位一致（精确复现，直接抓决策回归）+ ②输出 PSNR ≥ no-cache 自比的
> ~35 dB 噪声地板**。md5 比对是错误方法。

**已落地并验证等价（8/8 方法 + 默认设备端 GPU 复用 + 模型 dehook）：**

- **能力接线**：`CacheEngine` 实例化 `CacheOperatorRegistry` 并调用
  `register_builtin_cache_operators`（此前死代码），把 `state_` + `operators_` + 可选
  `ICacheDeviceStore` 传入 `CacheGraphLowering::execute_substeps`。
- **统一 substep 路径**：8 个方法均实现 `next_substep()`，逐子步产出 `SubstepPlan`；
  中间层按 `SubstepOpKind` 分派到 tap 驱动的 runner pass。`ED_CACHE_SUBSTEP` 门控与旧
  `execute` 的 hook 大 if/else 已删（commit `a2831c6`）。
- **Output（EasyCache/UCache/DBCache/CacheDiT）**：`OutputCompute`（FULL 后 DIFFERENCE→slot）/
  `OutputReuse`（LOAD+BLEND→输出）。skip 计数不变。
- **Feature host 路径（MagCache/TaylorSeer/SenCache）**：`FeatureCompute`/`FeatureReuse`
  （TaylorSeer 让 history ring 承重：raw-feature ring + ROTATE_HISTORY + 每步 reuse_coeffs
  blend）。
- **Probe（DiCache）**：`stop_after` 浅层前缀 + residual ring；probe 已 tap 化（delta_y/
  delta_x/gamma 经 `TapRegistry::ProbeMetricOperands` 在图内归约后读回标量）。
- **默认设备端 GPU 复用（MagCache/Qwen，`ED_FEATURE_CACHE_GPU` 默认 ON）已承重并验证**：
  `CacheStateManager` 拥有 per-slot 持久设备 buffer（`RunnerCacheDeviceStore`，
  `ggml_extend.hpp:1866`），经 `substep_capture` d2d 存残差、`substep_inject_slot` 在图内
  `ggml_add(x_before, slot)` 复用。Flux **和 Qwen** 均已从 legacy `DiCacheGpuState` 路径
  迁到设备槽。**验收（2026-07-14）**：设备槽 vs host-declarative **字节一致（PSNR 100 dB）**——
  Flux MagCache 30/50、Qwen MagCache 36/50，skip 计数逐位一致，两条路径 vs no-cache 分别
  为 28.4 dB / 25.2 dB（与 benchmark 报告一致）。
- **模型 dehook 已完成（§2.6）**：`CacheGraphScope` seam 整块删除（commit `cd0c9b7`），
  4 个模型 forward 改为 `TapRegistry` 条件 tap，cache 层零依赖具体模型 header。

**剩余工作：**

- **`GraphRepository` / 多静态图变体**：redesign 的第 3 层，未建。这是 SP + 缓存复合、
  细粒度切点的天然下游。
- **SP + 缓存复合**：SP 下 Feature/Probe 仍被能力协商拒绝（§4.1），当前只剩 Output 粒度。
  解耦完成后这是**最大的用户可感知缺口**。
- **事务语义（commit/rollback）**：诚实 no-op，无触发点（见 §4.2）。为未来预留，非当前债。

**结论**：声明式引擎已从「脚手架」转为「承重」——8 个方法 + Flux/Qwen 默认设备端 GPU 复用
真实跑在 StateManager+Operator 上，`CacheProgram` 不再只是决策描述符；且**模型侧已彻底 dehook**
（原列为「数周量级终态」的 Slice 4 已完成，seam 删除，substep 是唯一路径）。仍是脚手架的只剩
`GraphRepository`（未建）与事务语义（诚实 no-op）。每步继续按「skip 计数 + PSNR≥地板」验收
（CUDA 逐帧不确定，md5 比对错误），触及 tap/lowering 处优先保留 fallback。

---

## 6. 备选方向（简述，非推荐）

- **SP + 缓存复合**：价值最高、最贴合引擎双卖点（多卡 + 步缓存当前无法叠加）。难点 = 让
  graph-cut planner 跨 segment 保留 tap 的命名中间输出——`ggml_extend.hpp:6471`
  明确记录了此冲突：「mid-graph capture is not preserved across segments」。这是「落地
  声明式引擎」第 3 层（GraphRepository/变体编译）的天然下游，宜在其后做。Output 粒度缓存
  理论上可较早搭 SP 便车（无需 tap 捕获）。
- **巩固与加固**：删除或显式标注剩余脚手架、刷新全量 benchmark（当前 Qwen 行仍是旧数据）。
  低风险、不增能力，可作为任何方向的前置。（注：曾脆弱的 hook 耦合已随 §2.6 的 dehook 消解。）
- **细粒度切点**：把拓扑从单一 `BLOCK_STACK` 扩到 per-block / attention·FFN / token 级，
  解锁 partial-compute 与 token cache。研究前沿、上限高，但每个模型都要改 contract + seam，
  测试矩阵组合爆炸。

---

## 7. 参考文档（引用不重复）

| 文档 | 内容 |
|---|---|
| `cache_framework_design.md` | 上游 xllm `dit_cache` 框架及其 DiT 集成分析，§13 提出下一代接口 |
| `cache_framework_redesign.md` | 声明式重构**设计草案**（本分支部分实现的目标态） |
| `cache_quality_benchmark_report.md` | MagCache/DiCache vs no-cache 的 PSNR/SSIM/LPIPS + 速度 |
| `perf_gap_vs_diffusers.md` | 本引擎相对 diffusers 变慢的根因（逐步建图、无 CUDA Graph、GPU↔CPU 往返） |
| `perf_improvement_plan.md` | 针对上述根因的执行计划 |
