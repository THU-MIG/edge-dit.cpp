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
`CacheEngine`（相对 `main`：89 文件、+14k/-4.2k 行）。本文做两件事：

1. **盘点现状**：哪些能力真正落地、优点与差距分别是什么；
2. **给出前进目标**：把「落地声明式引擎」作为推荐方向深入分析其优势与问题，并列出备选方向。

本文以 `docs/cache_framework_redesign.md` 描述的声明式架构为**目标态**，衡量「已落地
vs 差距」。文中所有「未承重 / 死代码」类断言均已由代码检索核实，并附 `file:line`。

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

在 `cache_graph_lowering.cpp` 的 `execute` 中按 granularity 分流：

- **Output**：黑盒，只用 `hooks.full()` + 主机侧输出 diff/复用，SP 下也可用；
- **Feature**：用 block-stack seam 捕获/注入残差（MagCache、TaylorSeer、SenCache）；
- **Probe**：跑浅层前缀再决策（DiCache）。

### 2.5 性能优化真实有效

详见 `docs/cache_quality_benchmark_report.md`（H200 / 50 步 / Flux+Qwen）：

- on-GPU MagCache 特征复用、DiCache GPU 探针重写、`ED_CACHE_COMPILED_GRAPHS`
  build-once 图复用；
- skip 成本从 ~40ms/skip 降到 ~1ms，且验证 byte-identical，质量不变；
- Flux：MagCache 2.05x、DiCache 默认 1.24x（近无损，PSNR ~40）。

---

## 3. 优点

1. **解耦目标基本达成**：新增 cache 方法 = 写一个 `ICachePolicy`；新增模型 = 暴露一个
   `DiTModelCacheContract` + 一个 block-stack seam。摆脱了「模型数 × 方法数」的组合爆炸。
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
  `!can_attempt_graph_cut_segmented_compute()`（`ggml_extend.hpp:6309`），而后者在
  「任何 process group 使能」时即为 true —— 故纯 SP 也会关掉 seam，SP 下只剩 Output
  粒度。这是引擎两大卖点（多卡并行 + 步缓存）当前无法叠加的直接体现。
- **hook 耦合脆弱**：`gpu_metric`/`branch_key` 曾静默把所有 Feature cache（MagCache/
  TaylorSeer/SenCache）归零跑成 0/N（已修），说明 hook 表面 subtle、易回归。
- **cache 路径曾有 GPU 内存泄漏**（已修），长跑仍需 chunked 生成绕过。

### 4.2 脚手架层：搭好但未承重（均已 grep 核实）

设计文档最有雄心的三块声明式机器，目前只落地了**决策层**：

- **`CacheStateManager` 空转**：全库无 `read / write / rotate_history / read_history`
  调用（只有 `initialize / begin_step / commit_step` 被 engine 调用）；每个 policy 自持
  `std::unordered_map<const void*, ...> states_`（magcache/dicache/easycache/ucache/
  sencache/taylorseer/condition 均如此）。`commit_step / rollback_step` 是**显式
  no-op**（`cache_state_manager.cpp:105-116` 注释：宿主向量原地提交，无需 flush）——
  所谓「事务式状态 + 回滚」并未提供任何真实安全性。典型未接驱动：TaylorSeer 声明
  `history_depth = n_derivatives + 1`（`taylorseer_policy.cpp:44`），正是 ring buffer
  的用武之地，却自己存 `TaylorSeerState` 的导数历史。
- **`CacheOperatorRegistry` + 4 个 operator 是死代码**：`register_builtin_cache_operators`
  **全库无人调用**，registry 从未实例化；`IdentityOperator/DifferenceOperator/
  LinearPredictOperator/WeightedBlendOperator` 的 `apply_host` 从未被执行。真正的残差/
  预测/blend 数学内联在 policy（如 `magcache_policy.cpp:353` 直接读 `b.residual`）、
  lowering、pipeline 里。
- **无 `GraphRepository` / 多静态图变体**：`GraphRepository`、`GraphVariantKey`、
  `get_or_compile` 在 `src/`、`examples/` 及除 redesign 外的所有 docs 中**零命中**。
  设计文档核心思想「运行时在多张预编译静态图变体间选择」**没有实现**。实际执行是
  hook/回调式：`cache_graph_lowering.cpp` 按 `variant->kind` 分支去调
  `hooks.full/capture/inject/inject_gpu/probe`。因此 `CacheProgram`/`GraphVariantPlan`
  是薄「决策描述符」（告诉 lowering 调哪个 hook），而非编译产物；`SegmentPlan`/
  `CacheAction` 字段大部分是未消费的元数据。

> **小结**：声明式组件里，只有 `CacheProgram`/`GraphVariantPlan`（作为薄描述符）+
> `CacheRunnerHooks` lowering 是承重的。`CacheStateManager` 的 history/事务 API 和
> 整个 operator registry 是朝 redesign 目标搭建、但尚未承重的脚手架。

---

## 5. 推荐前进目标：落地声明式引擎

**方向**：把 §4.2 的三块脚手架真正接上，兑现 `cache_framework_redesign.md` 的核心承诺。

### 5.1 具体含义（分层，每层可独立 byte-verify against 当前 hook 路径）

1. **StateManager 承重化**：把 `CacheStateManager` 接为 TaylorSeer / DiCache / MagCache
   的真实状态后端（history ring + 真正的 commit/rollback），替换各 policy 的私有
   `states_`。
2. **Operator 化数学**：把残差 diff / 预测外推 / blend 路由到 `CacheOperator`，policy 只
   引用 operator id，`register_builtin_cache_operators` 真正被调用。
3. **GraphRepository + 变体编译**：引入真正的 FULL/REUSE/PREDICT 静态图变体编译与缓存，
   并与 `ED_CACHE_COMPILED_GRAPHS` 的 build-once 图复用融合（两者都想「少建图」）。

### 5.2 优势

- 兑现 redesign 承诺、偿还架构债；
- 为细粒度 / 轨迹修正 / token cache 打地基（这些方法都需要真实的多历史状态与 operator 组合）；
- 真正的事务状态让 rollback 安全（当前 no-op，异常步可能留下不一致历史）；
- operator 化让新方法接入更快（组合已有 operator，而非每次内联新数学）；
- 与 compiled-graph 复用合流，可能进一步降低建图开销。

### 5.3 问题 / 风险（不回避）

- **当前 hook 路径已够用**：纯「接脚手架」若不解锁具体能力，可能是**无用户可见收益**的
  工作量。因此本文主张——**把落地声明式引擎当作「手段」而非「目的」**：每一层都绑定一个
  具体收益（如 rollback 安全性、或为 SP / 细粒度铺路），按可 byte-verify 的最小切片推进，
  拒绝为「架构纯洁」而重写。
- **过度设计风险**：`GraphRepository` / 多变体在 GGML 静态图上与既有 `PlanCache` /
  `compute_reuse`（`ED_CACHE_COMPILED_GRAPHS`）职责重叠。落地前须先厘清边界，
  否则是两套「图缓存」互相打架（参见 `perf_gap_vs_diffusers.md` 的建图开销分析）。
- **回归风险高**：触及最脆弱的 seam / graph-cut 子系统与曾出事的 hook 耦合（§4.1）。
- **收益递延**：三层里只有第 3 层（GraphRepository）直接关联 SP 复合与细粒度这些
  用户可感知能力；前两层是纯内部整洁化，收益偏「未来价值」。

### 5.4 建议的首个最小切片

只把 `CacheStateManager` 接为 **TaylorSeer** 的 history 后端——它有最明确的多历史驱动
（`n_derivatives+1` 深度），改动面小、边界清晰。验收 = 输出对当前 hook 路径 **byte-identical**，
以此作为「脚手架可承重」的证明点，再决定是否推进第 2、3 层。**若这一层做完发现无实际收益，
即是「当前 hook 已够用」的有力信号，应及时止步而非硬推。**

### 5.5 实施进展（2026-07-13，声明式化已启动）

> **验收基准更正**：§5.1/§5.4 原写「byte-identical」，实测证伪——本引擎 CUDA 后端**逐帧
> 不确定**（同一 no-cache 配置连跑两次 PSNR ≈ 35 dB，非 100%）。因此等价性验收改为
> **①skip 计数逐位一致（精确复现，直接抓决策回归）+ ②输出 PSNR ≥ no-cache 自比的
> ~35 dB 噪声地板**。md5 比对是错误方法。

**已落地并验证等价（低风险地基）：**

- **能力接线**：`CacheEngine` 现实例化 `CacheOperatorRegistry` 并调用
  `register_builtin_cache_operators`（此前死代码），把 `state_` + `operators_` 传入
  `CacheGraphLowering::execute`。
- **per-method 开关**：`variant_has_actions(program)` —— 发出 `CacheAction` 的方法走声明式
  `ActionInterpreter`（LOAD/STORE/DIFFERENCE/BLEND over slot + operator）；未迁移方法走
  原 `reconstruct/observe` legacy 路径，逐字不变。可一次迁一个方法。
- **EasyCache / UCache**（Output）：`make_output_diff_program`（FULL.after
  DIFFERENCE→slot；REUSE.before LOAD+BLEND→输出）。skip 7/20、9/20 不变；diff 张量移入 slot。
- **MagCache**（Feature，host 路径）：`make_feature_reuse_program`（FULL.after STORE 残差
  →slot；REUSE.before LOAD→注入）。host/GPU 均 reuse 12/20，host-decl vs gpu-legacy 33 dB。
  seam 的 capture/inject 保留为**合法的模型 compute 边界**，只把残差**存储**移入 slot。
- **无回归**：DiCache 4/20、CacheDiT 1/20 不变；TaylorSeer 0/20 经 stash+重建证实为
  **既有**（默认 Flux 20 步 schedule 本就不触发），非本次迁移所致。

**剩余切片的风险评估（诚实）：**

- **3b TaylorSeer 导数**：非机械移植。导数级联 `dy[d+1]=(dy[d]−dy_prev[d])/window` 是**有状态
  递推**，且 window 随计算步间隔运行时变化；无法从 slot 里 K 个原始特征重建，也不匹配当前
  operator 的编译期 `params`。需要 IR 扩展（typed state-slot 或运行时参数绑定）。且默认设置
  skip 0/20，改动难以就地验证——建议连带 schedule 一起处理。
- **3c GPU operator lowering**：需给 `ICacheOperator` 增加**发射 ggml 节点**的 lowering（而非
  host `apply_host`），并把 seam `cache_graph_scope.hpp` 的 `add_injected` 硬编码残差重建
  改为 IR 驱动。触及最脆弱子系统，on-GPU 路径默认开启，高回归风险。
- **3d DiCache probe/gamma**：`build_probe_metrics`/gamma 轨迹对齐迁到 IR，依赖 3c。
- **Slice 4 彻底去 hook**：需为 flux/qwen/sd3/wan 各写 `IDiTGraphBuilder` 分段建图接口，
  让 seam 跨 graph-cut 存活——数周量级。

**结论**：声明式引擎已从「脚手架」转为「承重」——3 个方法真实跑在 StateManager+Operator 上，
`CacheProgram` 不再只是决策描述符。但「无 hook」的终态仍需上述 IR/seam 扩展；每一步继续按
「skip 计数 + PSNR≥地板」验收，逐切片推进，触及 seam 处优先保留 fallback。

---

## 6. 备选方向（简述，非推荐）

- **SP + 缓存复合**：价值最高、最贴合引擎双卖点（多卡 + 步缓存当前无法叠加）。难点 = 让
  graph-cut planner 跨 segment 保留 seam 的命名中间输出——`ggml_extend.hpp:6306-6308`
  明确记录了此冲突：「mid-graph capture is not preserved across segments」。这是「落地
  声明式引擎」第 3 层（GraphRepository/变体编译）的天然下游，宜在其后做。Output 粒度缓存
  理论上可较早搭 SP 便车（无需 seam）。
- **巩固与加固**：删除或显式标注脚手架、修复脆弱 hook 耦合、刷新全量 benchmark
  （当前 Qwen 行仍是旧数据）。低风险、不增能力，可作为任何方向的前置。
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
