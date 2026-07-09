# Flux SP 双卡加速比 Profile 与根因定位

这份文档记录当前 clean Flux Ulysses SP 的真实性能账本。目标是定位双卡加速比偏低的主矛盾，而不是继续做零散小算子优化。

当前边界：

```text
不改 CUDA/math 算子。
不改 collective 通信接口。
不把 RoPE、零散 reshape、小 fused op 当根因。
不把 sp:*attn_head_to_seq 这个 segment 名字直接等同于 head_to_seq helper 慢。
优先处理我们自己的 ggml graph / graph-cut 执行开销。
```

## 1. 当前结论

单卡基线：

```text
historical single GPU plain transformer warm avg ≈ 703.8ms

same-commit P0 rerun:
  log              = /tmp/flux_1gpu_plain_current_profile.log
  count            = 50
  warm excl step 1 = 707.9ms
  last10 avg       = 706.1ms
```

这个 P0 复跑和历史 703.8ms 只差约 0.6%，后面的根因判断不依赖这点波动。

clean 双卡 SP 修前：

```text
2GPU SP transformer warm avg ≈ 625.6ms
speedup ≈ 703.8 / 625.6 ≈ 1.12x

graph-cut total ≈ 612.1ms
compute         ≈ 508.5ms
comm            ≈  31.9ms
graph overhead  ≈  71.7ms
```

本轮已修两个明确的 graph overhead 点：

```text
1. pe runtime tensor 在 57 个 attention/head_to_seq segment 中重复 H2D 上传。
2. collect_future_input_names 每个 step 反复按 segment 向后扫描。
```

pe cache + collect_future 修后 1024x1024 / 50 step 双卡 SP：

```text
log:
  /tmp/flux_2gpu_sp_final_profile.log

2GPU SP transformer warm avg, excluding step 1:
  ≈ 606.8ms

last10 transformer avg:
  ≈ 605.7ms

speedup vs single plain:
  703.8 / 606.8 ≈ 1.16x
  707.9 / 606.8 ≈ 1.17x, using same-commit P0 rerun
```

修后 graph-cut warm avg：

```text
graph total   ≈ 592.9ms
compute       ≈ 508.7ms
comm          ≈  35.0ms
plan          ≈   8.0ms
build         ≈   3.3ms
alloc         ≈   7.4ms
copy          ≈   6.1ms
cache         ≈   7.2ms
other         ≈  17.1ms

graph overhead excluding compute + comm:
  592.9 - 508.7 - 35.0 ≈ 49.2ms
```

继续修掉两类 graph layout 重复 materialize 后：

```text
code changes:
  1. flux_sp_materialize_cut() 默认 clean path 不再对已 contiguous 的 q/k/v 额外 ggml_cont。
  2. sp_parallel helper 的通信 flatten 对已 contiguous tensor 改用 ggml_reshape_1d，
     不再用 ggml_cont_1d 复制一遍。

log:
  /tmp/flux_2gpu_sp_50_after_layout_flatten.log

2GPU SP transformer warm avg, excluding step 1:
  ≈ 597.0ms

last10 transformer avg:
  ≈ 599.2ms

speedup vs single plain:
  703.8 / 597.0 ≈ 1.18x
  707.9 / 597.0 ≈ 1.19x, using same-commit P0 rerun
```

当前 graph-cut warm avg：

```text
graph total   ≈ 583.2ms
compute       ≈ 498.5ms
comm          ≈  34.9ms
plan          ≈   7.7ms
build         ≈   3.4ms
alloc         ≈   7.2ms
copy          ≈   6.1ms
cache         ≈   7.9ms
other         ≈  17.3ms

graph overhead excluding compute + comm:
  583.2 - 498.5 - 34.9 ≈ 49.8ms
```

一句话判断：

```text
pe cache 和 collect_future plan-time 化拿回了约 18-20ms/step，
后续两个 graph layout 修复又拿回约 9-10ms/step。
但 clean SP 的主矛盾仍然是 compute ≈ 498.5ms 太高。
```

## 2. 本轮修复验证

### 2.1 pe runtime const device cache

修前 cache-off, 1024x1024 / 5 step：

```text
log:
  /tmp/flux_2gpu_sp_pe_cache_off_profile.log

step #5 overhead-detail:
  copy_tensor_set      ≈ 21.843ms
  copied_tensors       = 463
  copied_bytes         ≈ 252.64MiB
  runtime_const_hits   = 0
  runtime_const_uploads= 0
```

修后 cache-on, 1024x1024 / 5 step：

```text
log:
  /tmp/flux_2gpu_sp_pe_cache_pruned_profile.log

step #5 overhead-detail:
  copy_tensor_set      ≈ 3.757ms
  copied_tensors       = 406
  copied_bytes         ≈ 10.39MiB
  runtime_const_hits   = 56
  runtime_const_hit_bytes
                       ≈ 238.00MiB
  runtime_const_uploads= 1
  runtime_const_upload_bytes
                       ≈ 4.25MiB
```

解释：

```text
pe shape = [2, 2, 64, 4352], f32
pe bytes = 2 * 2 * 64 * 4352 * 4 = 4.25MiB

attention/head_to_seq 相关 segment 约 57 个:
  57 * 4.25MiB ≈ 242MiB

profile 中消失的 copied_bytes 正好对应这批重复 pe 上传。
```

实现边界：

```text
只缓存名为 pe 且 >= 1MiB 的 f32 runtime input。
cache entry 持有独立 ggml context + backend buffer，不依赖 segment-local graph 生命周期。
默认开启，可用 ED_RUNTIME_CONST_CACHE=0 关闭。
每个 step 只复用同一 host pointer 的 pe；遇到同名同 shape 新 host pointer 会清理旧 entry，避免 50 step 累积无用 device cache。
```

### 2.2 collect_future_input_names plan-time 化

修前：

```text
collect_future ≈ 5.7-5.8ms / step
```

修后 1024x1024 / 50 step：

```text
overhead warm avg:
  collect_future ≈ 0.005ms

overhead last10 avg:
  collect_future ≈ 0.004ms
```

实现：

```text
Plan::Segment 增加 future_input_names。
plan build / max-vram merge 后倒序预计算每个 segment 的 future previous-cut input name set。
compute_with_graph_cuts 热路径直接引用 segment.future_input_names。
```

这避免了每个 step 里 193 个 segment 反复向后扫描 future segments 的 O(N^2) bookkeeping。

## 3. 修后仍然不够快的原因

目标时间预算：

```text
1.3x target: 703.8 / 1.3 ≈ 541.4ms
1.4x target: 703.8 / 1.4 ≈ 502.7ms
1.5x target: 703.8 / 1.5 ≈ 469.2ms
1.6x target: 703.8 / 1.6 ≈ 439.9ms
```

当前修后：

```text
transformer warm avg ≈ 606.8ms
graph total          ≈ 592.9ms
compute              ≈ 508.7ms
comm                 ≈  35.0ms
graph overhead       ≈  49.2ms
```

即使把剩余 graph overhead 从 49ms 再压到 25ms，且通信保持 35ms：

```text
ideal-ish total ≈ 508.7 + 35 + 25
                ≈ 568.7ms

speedup ≈ 703.8 / 568.7 ≈ 1.24x
```

这说明：

```text
继续只优化 runner overhead 已经不可能把 clean SP 推到 1.4x/1.5x。
compute bucket 必须下降。
```

1.5x 的 compute 预算大致是：

```text
target total ≈ 469ms
假设 comm ≈ 30-35ms
假设 graph overhead ≈ 25-35ms

compute budget ≈ 469 - 30~35 - 25~35
               ≈ 399-414ms
```

当前：

```text
compute ≈ 508.7ms
```

所以 compute 还需要减少约：

```text
508.7 - 399~414 ≈ 95-110ms
```

这才是当前 clean SP 加速比偏低的主矛盾。

## 4. 当前 overhead 账本

修后 50 step warm avg：

```text
copy             ≈ 6.1ms
cache            ≈ 7.2ms
alloc            ≈ 7.4ms
plan + build     ≈ 11.3ms
other            ≈ 17.1ms
collect_future   ≈ 0.005ms
copy_tensor_set  ≈ 3.79ms
copied_bytes     ≈ 10.39MiB
```

已解决或大幅降低：

```text
pe 重复上传:
  copied_bytes 252.64MiB -> 10.39MiB
  copy_tensor_set 21.8ms -> 3.8ms

collect_future:
  5.8ms -> 0.005ms
```

仍可优化但不是决定性主矛盾：

```text
plan + build ≈ 11ms
alloc        ≈ 7ms
cache        ≈ 7ms
other        ≈ 17ms
```

这些加起来还有价值，但即使全砍掉一半，也不能解释 1.5x 所需的 95-110ms compute 缺口。

## 5. 现在可以排除的方向

当前证据排除：

```text
1. 不要优先改 NCCL / collective。
   comm ≈ 35ms，不是主瓶颈。

2. 不要继续优化 RoPE 算子。
   pe 问题是 runtime input 重复上传，不是 RoPE math 慢。

3. 不要继续盯 head_to_seq pack/unpack 小 helper。
   之前 hard-boundary profile 已显示 head_to_seq helper 本身不是 100ms 级大头。

4. 不要只看 segment 名字判断瓶颈。
   sp:*attn_head_to_seq 不是纯 head_to_seq helper。

5. 不要为了省 copy 简单 alias recv placeholder。
   graph-cut comm output materialization 语义不能破坏。

6. 不要删 double->single gather/concat/resplit。

7. 不要删 final gather。

8. 不要做 async overlap。
```

## 6. Compute root-cause profile

已经新增 profile-only 的 compute bucket 归因：

```text
开关:
  ED_PROFILE_GRAPH_CUTS=1
  ED_PROFILE_GRAPH_CUTS_TOP=0
  ED_PROFILE_GRAPH_CUTS_COMPUTE_TOP=24

log:
  /tmp/flux_2gpu_sp_compute_root_profile.log
```

这个开关会在 graph-cut profile 中额外输出：

```text
compute-bucket:
  按 Flux SP segment 语义粗聚合 compute/comm/copy/cache。

top-compute:
  按 segment compute_ms 排序，输出 segment name、bucket、node count、op histogram、输入/输出 shape 摘要。

op histogram:
  统计 segment 原始 graph 内部 node op 类型。
  重点区分 MUL_MAT/FLASH_ATTN/RMS_NORM/ADD/MUL 等 math ops，
  以及 CONT/CPY/RESHAPE/PERMUTE/VIEW/CONCAT 等 layout/materialize ops。
```

实现边界：

```text
不设置 ED_PROFILE_GRAPH_CUTS_COMPUTE_TOP 时不采集 op histogram，不输出 compute-bucket/top-compute。
普通 ED_PROFILE_GRAPH_CUTS=1 的 overhead profile 不会被这个工具额外污染。
```

### 6.1 P0 单卡基线复核

同 commit 单卡 1024x1024 / 50 step：

```text
log:
  /tmp/flux_1gpu_plain_current_profile.log

transformer elapsed:
  count            = 50
  all avg          = 708.4ms
  warm excl step 1 = 707.9ms
  last10 avg       = 706.1ms
```

结论：

```text
历史 703.8ms 单卡基线仍然有效。
当前复跑约 706-708ms，只是正常 run-to-run 波动。
```

### 6.2 SP compute bucket 第一层归因

短跑 1024x1024 / 3 step 的第 3 个 denoise step：

```text
graph cut profile #3:
  total   = 732ms
  compute = 505ms
  comm    =  89ms
```

这个短跑开了比较重的 top-compute shape 日志，且 step 数少，所以 total/comm/alloc 不作为最终性能基线；但 graph compute bucket 的组成已经很清楚。

第 3 step compute bucket：

| bucket | segments | compute | comm | math_ops | layout_ops | 主要 ops |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sp.single.attn_head_to_seq` | 38 | **172ms** | 9ms | 532 | **2090** | `CONT:722, RESHAPE:532, PERMUTE:342, VIEW:304` |
| `flux.single_blocks.tail` | 38 | **94ms** | 0ms | 1254 | 503 | `ADD:532, MUL_MAT:342, UNARY:190` |
| `sp.double.attn_head_to_seq` | 19 | **90ms** | 6ms | 342 | **1520** | `CONT:589, VIEW:342, RESHAPE:228, PERMUTE:152, CONCAT:114` |
| `sp.single.qkv_seq_to_head` | 38 | **79ms** | 19ms | 1064 | 921 | `ADD:456, VIEW:305, MUL_MAT:304, CONT:229` |
| `sp.double.qkv_seq_to_head` | 38 | **35ms** | 54ms | 1068 | 923 | `ADD:458, MUL_MAT:306, VIEW:306, CONT:232` |
| `flux.double_blocks.tail` | 19 | **34ms** | 0ms | 1049 | 543 | `ADD:458, MUL_MAT:268, RESHAPE:193` |

最关键的合计：

```text
attn_head_to_seq buckets:
  sp.single.attn_head_to_seq compute = 172ms
  sp.double.attn_head_to_seq compute =  90ms
  total                         = 262ms

262ms / 505ms compute ≈ 51.9%
```

也就是说，当前 compute bucket 里最大头不是 MLP tail，也不是 qkv seq_to_head，而是 `attn_head_to_seq` 这一类 segment。

但这里的含义要非常小心：

```text
这不是说 all-to-all 通信慢。
这不是说 head_to_seq recv 慢。
这也不等同于某个小 helper pack/unpack 慢。

op histogram 说明这些 segment 里主要是 layout/materialize graph work：
  CONT / RESHAPE / PERMUTE / VIEW / CONCAT
```

典型 top-compute segment：

```text
sp:flux_single*_attn_head_to_seq
  nodes      = 69
  math_ops   = 14
  layout_ops = 55
  ops        = CONT:19, RESHAPE:14, PERMUTE:9, VIEW:8,
               MUL:6, REPEAT:4, ADD:2, RMS_NORM:2, MUL_MAT:2

sp:flux_double*_txt_img_attn_head_to_seq
  nodes      = 98
  math_ops   = 18
  layout_ops = 80
  ops        = CONT:31, VIEW:18, RESHAPE:12, PERMUTE:8,
               CONCAT:6, MUL:8, REPEAT:4, RMS_NORM:4, MUL_MAT:2
```

这比之前的结论更具体：

```text
clean SP compute 没降下来，第一层主因是 attn_head_to_seq 相关 segment
包含大量 layout/materialize graph work。

这类 work 被记在 ggml_backend_graph_compute() 里，
所以它不会显示为 NCCL comm，也不会显示为 runner overhead。
```

### 6.3 对优化方向的直接影响

现在下一刀不该是：

```text
RoPE
NCCL
pe cache
collect_future
简单 head_to_seq micro-op
async overlap
```

现在应该沿着这个方向查：

```text
1. 为什么 attn_head_to_seq segment 包含这么多 CONT/RESHAPE/PERMUTE/VIEW/CONCAT？

2. 哪些 layout op 是 graph-cut boundary 强制 materialize？

3. 哪些 layout op 来自 SP layout 变换本身，可以通过 graph 组织减少？

4. 是否可以把 attention core 与 head_to_seq 后处理重新切 boundary，
   避免每个 block 的 post-attention path 被迫落成大量小 layout graph。

5. 是否存在可以在 ggml graph 层去掉的重复 contiguous/permute/reshape 链，
   而不改 CUDA kernel、不改 collective 接口、不破坏 comm output materialization 语义。
```

## 7. Materialize Stage Profile

第一层 compute bucket 已经说明 `attn_head_to_seq` 是最大 compute 来源；第二层 profile 进一步回答：

```text
这些 CONT / CONCAT / DUP 到底来自哪段语义路径？
是通信 input/output，还是 qkv recv 恢复、attention q/k/v layout、double txt/img concat、tail path？
```

新增 profile-only 开关：

```text
ED_PROFILE_GRAPH_CUTS_MATERIALIZE_TOP=N
```

它会额外输出：

```text
materialize-bucket:
  按 bucket 聚合 CONT / CPY / CONCAT / DUP 的 logical materialize bytes。

stages:
  按语义阶段聚合 materialize bytes，例如：
    qkv.recv_output_restore
    attention.qkv_layout
    attention.rope_qk_layout
    double.attn_qkv_concat
    head_to_seq.recv_output_restore
    qkv.send_combine
    qkv.input_view_materialize
    comm.recv_placeholder

top-materialize:
  对 top compute segment 输出最重 materialize node 的 shape、stride、producer、consumer、boundary/comm flags。
```

实现边界：

```text
默认不启用。
不改 graph 数学语义。
不改 CUDA/math kernel。
不改 collective 接口。
不改变 graph-cut comm output materialization 语义。
```

本节数据来自 clean build：

```text
binary:
  ./build-cuda-clean/bin/ed-cli

log:
  /tmp/flux_2gpu_sp_materialize_stage_profile.log

output:
  /tmp/flux_2gpu_sp_materialize_stage_profile.png

sha256:
  60b7568181785a32da06748fe2ce718dee33b0032940c0bc49a37025b8235fc9
```

该 hash 与已知 good 输出一致：

```text
/tmp/flux_2gpu_sp_materialize_flatten_reshape.png
  60b7568181785a32da06748fe2ce718dee33b0032940c0bc49a37025b8235fc9
```

注意：

```text
旧的 /tmp/flux_2gpu_sp_materialize_chain_profile.log 不能再作为 clean 结论使用。
该日志来自旧 binary / 旧口径，且 build-cuda 曾带 ED_DEBUG_SP_COMM=ON。
当前 clean 结论以 build-cuda-clean + stage profile 为准。
```

### 7.1 已完成的两类 graph layout 修复

在 stage profile 前，已经完成两类不会改变数学语义的 graph layout 修复：

```text
1. flux_sp_materialize_cut():
   默认 clean path 不再对已经 contiguous 的 q/k/v 额外 ggml_cont。
   strict barrier 打开时仍保留 materialize + graph cut 语义。

2. sp_parallel helper:
   通信 flatten 对已 contiguous tensor 改用 ggml_reshape_1d。
   不再用 ggml_cont_1d 复制一遍。
```

A/B 后的关键变化：

| bucket | before materialize | after materialize | before cont | after cont | compute |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp.single.attn_head_to_seq` | 14858.00MiB | **10982.00MiB** | 13889.00MiB | **10013.00MiB** | 171ms -> 168ms |
| `sp.double.attn_head_to_seq` | 10834.75MiB | **8412.25MiB** | 7913.50MiB | **5491.00MiB** | 99ms -> 96ms |
| `sp.single.qkv_seq_to_head` | 16652.84MiB | **13745.84MiB** | 8747.84MiB | **5840.84MiB** | 88ms -> 82ms |
| `sp.double.qkv_seq_to_head` | 8266.67MiB | **6813.17MiB** | 4390.67MiB | **2937.17MiB** | 52ms -> 51ms |

端到端 50 step clean SP：

```text
log:
  /tmp/flux_2gpu_sp_50_after_layout_flatten.log

transformer warm avg, excluding step 1:
  606.8ms -> 597.0ms

graph compute:
  508.7ms -> 498.5ms
```

这说明这两刀有效，但不是根因级解决：

```text
拿回约 9-10ms。
当前 speedup 仍只有约 1.18x-1.19x。
剩余 compute ≈ 498.5ms 仍然过高。
```

### 7.2 当前 clean materialize bucket

已修复后，clean build 当前 materialize bucket：

| bucket | materialize | CONT | CONCAT | DUP | 关键 stage |
| --- | ---: | ---: | ---: | ---: | --- |
| `sp.single.attn_head_to_seq` | **10982.00MiB** | **10013.00MiB** | 0.00MiB | 969.00MiB | qkv layout / rope qk layout / qkv recv restore |
| `flux.single_blocks.tail` | **9869.84MiB** | 4871.84MiB | **4947.00MiB** | 51.00MiB | linear2 input concat / qkv input view |
| `sp.double.attn_head_to_seq` | **8412.25MiB** | **5491.00MiB** | **2436.75MiB** | 484.50MiB | double txt/img qkv concat |
| `sp.single.qkv_seq_to_head` | **13745.84MiB** | 5840.84MiB | 4947.00MiB | 2958.00MiB | qkv send combine / qkv input view |
| `sp.double.qkv_seq_to_head` | **6813.17MiB** | 2937.17MiB | 2422.50MiB | 1453.50MiB | qkv send combine / qkv input view |

最重要的变化：

```text
comm_in materialize 已经降为 0。
cont_from_cont 已经降为 0。

说明前一轮修复确实消掉了通信输入 flatten 和明显 CONT->CONT。
剩下的大头不是 send_flat，也不是 recv placeholder。
```

### 7.3 attn_head_to_seq Stage Breakdown

`sp.single.attn_head_to_seq` 当前分解：

| stage | bytes | CONT | 含义 |
| --- | ---: | ---: | --- |
| `attention.qkv_layout` | **2907.00MiB** | 2907.00MiB | q/k/v 进入 attention 前的 layout 转换 |
| `attention.rope_qk_layout` | **1938.00MiB** | 1938.00MiB | q/k RoPE 内部第二段 layout 转换 |
| `qkv.recv_output_restore` | **1938.00MiB** | 1938.00MiB | q/k 从 qkv_seq_to_head recv_flat 恢复成输出 layout |
| `permute_to_cont_layout` | 1292.00MiB | 1292.00MiB | 零散 permute/view 后接 CONT |
| `attention.output_split_head` | 969.00MiB | 969.00MiB | attention 输出拆回 head layout |
| `attention.qkv_from_boundary` | 969.00MiB | 969.00MiB | v 从 boundary input 恢复 |
| `comm.recv_placeholder` | 969.00MiB | 0.00MiB | graph-cut comm output placeholder |

`sp.double.attn_head_to_seq` 当前分解：

| stage | bytes | CONT | CONCAT | 含义 |
| --- | ---: | ---: | ---: | --- |
| `double.attn_qkv_concat` | **3876.00MiB** | 2422.50MiB | 1453.50MiB | txt/img q/k/v concat 后又进入 attention layout |
| `qkv.recv_output_restore` | **1453.50MiB** | 1453.50MiB | 0.00MiB | txt/img qkv_seq_to_head recv 输出恢复 |
| `concat_layout` | 983.25MiB | 0.00MiB | 983.25MiB | 其它 concat 链 |
| `attention.output_split_head` | 969.00MiB | 969.00MiB | 0.00MiB | attention 输出拆回 txt/img head layout |
| `permute_to_cont_layout` | 646.00MiB | 646.00MiB | 0.00MiB | 零散 permute/view 后接 CONT |
| `comm.recv_placeholder` | 484.50MiB | 0.00MiB | 0.00MiB | graph-cut comm output placeholder |

这给出更具体的判断：

```text
sp.*attn_head_to_seq 这个 bucket 名字仍然容易误导。

它里面的大头不是 head_to_seq all-to-all。
也不是 recv placeholder。

single 最大是 q/k/v attention layout 和 q/k RoPE 前后 layout。
double 最大是 txt/img q/k/v concat 后又立刻进入 attention layout。
```

### 7.4 qkv_seq_to_head 与 Tail 也需要看

`sp.single.qkv_seq_to_head` 当前 stage：

| stage | bytes | 含义 |
| --- | ---: | --- |
| `qkv.send_combine` | **5814.00MiB** | q/k/v input 合并后 pack 到 seq_to_head all-to-all |
| `qkv.input_view_materialize` | **4845.00MiB** | 从 qkv_mlp / qkv projection view 出 q/k/v 后 materialize |
| `comm.recv_placeholder` | 2907.00MiB | qkv recv placeholder |

`sp.double.qkv_seq_to_head` 当前 stage：

| stage | bytes | 含义 |
| --- | ---: | --- |
| `qkv.send_combine` | **2907.00MiB** | txt/img q/k/v send combine |
| `qkv.input_view_materialize` | **2448.00MiB** | qkv projection view materialize |
| `comm.recv_placeholder` | 1453.50MiB | qkv recv placeholder |

`flux.single_blocks.tail` 当前 stage：

| stage | bytes | 含义 |
| --- | ---: | --- |
| `concat_layout` | **4972.50MiB** | `attn_flat + mlp` 进入 `linear2` 前 concat |
| `qkv.input_view_materialize` | **3876.00MiB** | single block qkv_mlp 的 MLP branch view materialize |
| `head_to_seq.recv_output_restore` | 969.00MiB | head_to_seq 输出作为后续 tail 输入恢复 |

所以剩余 compute 大桶已经不是单一问题，而是两个层次：

```text
P0: attention 前后的 SP layout restore / concat / RoPE layout 链。
P1: qkv_seq_to_head send combine 与 qkv_mlp / MLP branch view materialize。
```

### 7.5 当前最具体的 Root Cause

当前 root cause 可以从：

```text
attn_head_to_seq 相关 segment layout/materialize tax 很大
```

进一步收敛成：

```text
SP helper 把 q/k/v 从 seq layout 变到 head layout 后，
Flux attention 又马上要求另一套 attention/RoPE layout。

于是每个 block 里出现：
  qkv recv_flat -> output restore CONT
  q/k/v attn layout PERMUTE(view)->CONT
  q/k RoPE 内部 RESHAPE/PERMUTE(view)->CONT
  attention output split/head_to_seq restore CONT

double stream 还多一层：
  txt q/k/v + img q/k/v -> CONCAT -> PERMUTE(view)->CONT

single tail 还多一层：
  attn_flat + mlp -> CONCAT -> linear2
```

这不是 NCCL，也不是 runner overhead，而是 graph 构造层在布局契约上没有对齐：

```text
view/reshape/permute 本身可能只是 metadata，
但每次后面接 CONT 就变成实际 device materialize。
```

### 7.6 下一刀方向

下一刀不要写成“优化 RoPE 算子”。更准确的目标是：

```text
让 SP helper 输出 layout 更贴近 Flux attention/RoPE 消费 layout，
避免先恢复成 [head_dim, shard_heads, seq]，
再马上 PERMUTE/CONT 成 [head_dim, seq, shard_heads]。
```

第一阶段建议只做 graph-level layout contract 实验：

```text
1. 给 q/k 的 seq_to_head 输出增加 seq-major variant:
   [head_dim, full_seq, shard_heads, batch]

2. 给 Flux SP attention 增加只用于该 layout 的 q/k RoPE wrapper:
   跳过 Rope::apply_rope() 的第一段 PERMUTE(view)->CONT。

3. v 仍保持 attention 当前需要的 [head_dim, shard_heads, full_seq, batch]，
   不改 attention kernel，不改 collective，不改 recv placeholder。

4. 先只用环境变量开关控制，例如：
   ED_FLUX_SP_QK_SEQ_MAJOR=1
```

验收先看 materialize stage，而不是先看端到端：

```text
ED_PROFILE_GRAPH_CUTS_MATERIALIZE_TOP=8

期望下降：
  sp.single.attn_head_to_seq:
    attention.qkv_layout
    qkv.recv_output_restore

  sp.double.attn_head_to_seq:
    qkv.recv_output_restore
    double.attn_qkv_concat 中 q/k 部分

必须不变或只小幅变化：
  comm.recv_placeholder
  final image hash
```

这一步如果成功，才继续处理：

```text
1. double txt/img q/k/v concat layout。
2. qkv_seq_to_head q/k/v send combine。
3. single tail 的 attn_flat + mlp concat / MLP branch view materialize。
```

## 8. 推荐命令

clean SP 50 step profile：

```bash
ED_RUNTIME_CONST_CACHE=1 \
ED_PROFILE_FLUX=1 \
ED_PROFILE_GRAPH_CUTS=1 \
ED_PROFILE_GRAPH_CUTS_TOP=0 \
./build-cuda-clean/bin/ed-cli --backend cuda --devices 3,5 \
  --model /path/to/flux-dev \
  -p "a cute cat" -W 1024 -H 1024 --steps 50 -s 0 --guidance 3.5 \
  --sp-size 2 \
  -o /tmp/flux_2gpu_sp_50_after_layout_flatten.png \
  2>&1 | tee /tmp/flux_2gpu_sp_50_after_layout_flatten.log
```

materialize stage profile：

```bash
ED_RUNTIME_CONST_CACHE=1 \
ED_PROFILE_GRAPH_CUTS=1 \
ED_PROFILE_GRAPH_CUTS_TOP=0 \
ED_PROFILE_GRAPH_CUTS_MATERIALIZE_TOP=8 \
./build-cuda-clean/bin/ed-cli --backend cuda --devices 3,5 \
  --model /path/to/flux-dev \
  -p "a cute cat" -W 1024 -H 1024 --steps 1 -s 0 --guidance 3.5 \
  --sp-size 2 \
  -o /tmp/flux_2gpu_sp_materialize_stage_profile.png \
  2>&1 | tee /tmp/flux_2gpu_sp_materialize_stage_profile.log
```

compute root-cause profile：

```bash
ED_RUNTIME_CONST_CACHE=1 \
ED_PROFILE_GRAPH_CUTS=1 \
ED_PROFILE_GRAPH_CUTS_TOP=0 \
ED_PROFILE_GRAPH_CUTS_COMPUTE_TOP=24 \
./build-cuda-clean/bin/ed-cli --backend cuda --devices 3,5 \
  --model /path/to/flux-dev \
  -p "a cute cat" -W 1024 -H 1024 --steps 3 -s 0 --guidance 3.5 \
  --sp-size 2 \
  -o /tmp/flux_2gpu_sp_compute_root_profile.png \
  2>&1 | tee /tmp/flux_2gpu_sp_compute_root_profile.log
```

single GPU P0 baseline：

```bash
ED_PROFILE_FLUX=1 \
./build-cuda-clean/bin/ed-cli --backend cuda --devices 5 \
  --model /path/to/flux-dev \
  -p "a cute cat" -W 1024 -H 1024 --steps 50 -s 0 --guidance 3.5 \
  -o /tmp/flux_1gpu_plain_current_profile.png \
  2>&1 | tee /tmp/flux_1gpu_plain_current_profile.log
```

关键字段：

```text
graph cut profile:
  total / compute / comm / alloc / copy / cache / other

materialize-bucket:
  materialize_bytes
  cont / concat / dup
  stages

top-materialize:
  stage_details
  nodes
  src_boundary_in / comm_in / comm_out flags

compute-bucket:
  bucket
  compute
  comm
  math_ops
  layout_ops
  ops
```

## 9. 最终判断

当前 clean 2GPU Ulysses SP 加速比从约 1.12x 提升到约 1.18x-1.19x：

```text
pe cache + collect_future:
  625.6ms -> 606.8ms

skip redundant materialize + flatten reshape:
  606.8ms -> 597.0ms
```

这说明本轮 graph overhead / graph layout 修复有效，但还不是根因级解决。

最重要的事实已经更新为：

```text
graph compute ≈ 498.5ms
```

它仍然几乎等于 1.4x 目标总时间：

```text
1.4x target total ≈ 502.7ms
```

因此后续优化重点必须继续在 graph compute 内部：

```text
不是通信慢。
不是 pe/RoPE math 慢。
不是 collect_future 慢。
不是 recv placeholder 慢。

当前更具体的主矛盾是：
  SP q/k/v layout 与 Flux attention/RoPE 消费 layout 不匹配，
  造成每个 block 内 qkv recv restore、attention q/k/v layout、
  q/k RoPE layout、double txt/img concat、tail concat 等 GiB 级 materialize。
```

下一刀应该是 graph-level layout contract 优化，而不是 CUDA kernel / NCCL / async overlap。

## 10. 本轮推进记录：layout contract 实验与被否方向

### 10.1 q/k seq-major layout contract 是正确的小收益

本轮增加了一个 gated graph-layout 实验：

```text
ED_FLUX_SP_QK_SEQ_MAJOR=1
```

语义是让 `seq_to_head_batched` 对 q/k 输出：

```text
[head_dim, full_seq, shard_heads, batch]
```

而 v 仍保持旧布局：

```text
[head_dim, shard_heads, full_seq, batch]
```

这样 Flux SP attention 里的 q/k RoPE wrapper 可以跳过 `Rope::apply_rope()` 入口处第一段 `PERMUTE(view)->CONT`。这不是 RoPE 数学优化，也不改 attention kernel，只是把 SP helper 的输出 layout contract 调整到下游实际消费 layout。

正确性：

```text
/tmp/flux_2gpu_sp_qk_seq_major_axis_fixed_stage_profile.png
/tmp/flux_2gpu_sp_materialize_stage_profile.png

sha256 = 60b7568181785a32da06748fe2ce718dee33b0032940c0bc49a37025b8235fc9
```

CPU 对拍：

```text
sp_parallel_test CPU world_size=2 passed
```

1-step materialize stage A/B：

| bucket | baseline materialize | q/k seq-major materialize | baseline CONT | q/k seq-major CONT | compute delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sp.single.attn_head_to_seq` | 10982.00MiB | 9044.00MiB | 10013.00MiB | 8075.00MiB | 169ms -> 161ms |
| `sp.double.attn_head_to_seq` | 8412.25MiB | 7443.25MiB | 5491.00MiB | 4522.00MiB | 97ms -> 92ms |

50-step profile：

```text
ED_FLUX_SP_QK_SEQ_MAJOR=1:
  transformer elapsed avg_all      = 607.06ms
  transformer elapsed avg_skip1    = 582.80ms
  graph total avg_skip1            = 568.69ms
  graph compute avg_skip1          = 485.57ms
  graph comm avg_skip1             = 35.57ms
  graph overhead avg_skip1         = 47.55ms
```

对比之前 clean fixed 约 `compute ~= 498.5ms`，这刀大约省 `10-15ms/step` compute。结论：

```text
q/k seq-major 是正确的 graph layout contract 小收益，证明 layout/materialize bucket 方向成立；
但它不是根因级突破，不能单独把双卡 SP 推到 1.4x/1.5x。
```

因此该路径应保持 gated，后续可在更多 seed/resolution/backend 验证后再考虑默认打开。

### 10.2 single tail split-linear 被 hash 否掉，已撤回

针对 `flux.single_blocks.tail` 的大桶：

```text
concat_layout ~= 4972.5MiB
```

尝试过一个 graph rewrite：

```text
linear2(concat(attn_flat, mlp))
  -> linear2_attn(attn_flat) + linear2_mlp(mlp) + bias
```

这个实验确实把 `flux.single_blocks.tail` 的 materialize 降低：

```text
materialize: 9869.84MiB -> 5024.84MiB
concat:      4947.00MiB -> 102.00MiB
concat_layout stage: 4972.5MiB -> 127.5MiB
```

但 Flux 输出 hash 不一致：

```text
baseline hash:     60b7568181785a32da06748fe2ce718dee33b0032940c0bc49a37025b8235fc9
split-linear hash: f7b00f79d303fcf95dd3179617762b08d8017a2a6d0c0fb96eb5616e7e07a677
```

原因不是数学公式错误，而是执行顺序从一个 `MUL_MAT` 改成两个 `MUL_MAT + ADD`，浮点累加顺序改变，无法满足当前 bitwise correctness 要求。同时 compute bucket 还变高：

```text
flux.single_blocks.tail compute: 98ms -> 110ms
```

结论：

```text
single tail concat 是真大桶，但不能用 split-matmul 这种改变累加顺序的 graph rewrite 交付。
该实验代码已撤回，不应继续沿这个方向推进，除非未来明确放宽 bitwise hash 要求或实现等价 fused concat-linear kernel。
```

### 10.3 直接跳过 seq_to_head input cont 被 CPU layout 对拍否掉，已撤回

另一个看起来低风险的想法是：

```text
sp_all_to_all_4d_seq_to_head_batched:
  q/k/v 是 qkv_mlp 或 qkv projection 的 view
  不先 ggml_cont(q/k/v)
  直接 ggml_concat(q, k, v, dim0)
```

目标是减少：

```text
sp.single.qkv_seq_to_head qkv.input_view_materialize ~= 4845MiB
sp.double.qkv_seq_to_head qkv.input_view_materialize ~= 2448MiB
```

但 CPU packed-qkv view 对拍证明，直接 concat 非连续 q/k/v view 会改变 `send_flat` 元素顺序：

```text
rank0 mismatch example:
  got      = 10001
  expected = 11
```

结论：

```text
qkv.input_view_materialize 不能靠简单跳过 input cont 解决。
当前 helper 的 input cont 是为了建立正确 send_flat layout，不是可直接删除的冗余 op。
```

这个失败实验也已撤回。测试中保留了 packed qkv view 输入的对拍，用来保护当前正确 layout 语义。

### 10.4 下一步更准确的方向

本轮两个 negative result 说明：

```text
1. 不能用改变浮点累加顺序的 split-matmul 绕开 tail concat。
2. 不能直接跳过 q/k/v view materialize，让 concat 自己处理非连续 view。
```

下一步仍然应该打 graph compute/layout 大桶，但要满足更严格约束：

```text
必须保持 send_flat / recv output layout 完全一致；
必须保持 Flux output hash 一致；
不能依赖 non-contiguous view concat 的隐式顺序；
不能改变 matmul 累加顺序；
不改 third_party/ggml，不改 CUDA/math kernel，不改 collective interface。
```

因此可行方向变成：

```text
P0. 继续完善 q/k seq-major layout contract，并验证 50-step / 多 seed / NCCL。
P1. 针对 qkv.input_view_materialize 做专门的等价 pack/layout 构造，而不是简单跳过 cont。
P2. 针对 double txt/img q/k concat，寻找不改变 attention 输入数值顺序的 layout contract。
P3. single tail concat 若要继续打，必须是等价 fused concat-linear 或保持单个 matmul 的输入 materialization 语义；简单 split-linear 不可用。
```

当前最值得做的下一刀仍然是 qkv path，但应该表述为：

```text
设计一个等价的 qkv_seq_to_head pack/layout 路径，直接从 qkv/qkv_mlp packed layout 生成当前 send_flat 语义，
而不是让三个 q/k/v view 各自 materialize 后再 concat。
```

这属于 graph/helper 层的 fused pack/layout 设计，不是 NCCL、async overlap、RoPE math 或 ggml CUDA kernel 优化。

## 11. 和正常 Ulysses / Torch SP 的异常对账

这一节只回答一个问题：

```text
当前 ggml graph-cut SP 和一个正常 Ulysses/Torch SP 相比，哪一块明显不正常、明显慢？
```

结论先说：

```text
不是某个数学算子本身异常慢。
不是 NCCL/all-to-all 本身异常慢。
最异常的是 attention 前后的 layout/materialize 链。

如果必须点名 op 类型：
  P0: GGML_OP_CONT
  P1: GGML_OP_CONCAT
  P2: graph-cut comm placeholder / boundary restore 诱发的 DUP + restore

如果必须点名代码区域：
  sp.single.attn_head_to_seq
  sp.double.attn_head_to_seq
```

### 11.1 正常 Ulysses/Torch SP 应该是什么账

正常 Ulysses 双卡 SP 的主流程应该接近：

```text
local sequence 上做 qkv / mlp / norm / elementwise
seq_to_head all-to-all
attention core 在 head-sharded layout 上运行
head_to_seq all-to-all
local sequence 上做 post-attn / tail
```

它可以有通信 pack/unpack，也可以有少量 layout conversion，但不应该出现：

```text
每个 block 反复把 q/k/v 从 recv layout materialize 成另一个 layout；
RoPE/attention 再把 q/k/v materialize 成第三种 layout；
attention output 再 materialize 回 head_to_seq 输入 layout；
double stream txt/img q/k/v concat 后马上又 permute/cont；
single tail 为 concat/linear2 产生 GiB 级 materialize。
```

Torch 后端通常不会把这些都显式落成大量独立 `contiguous/cat/permute` kernel：

```text
1. torch tensor view/permute 本身多是 metadata；
2. SDPA/FlashAttention 通常接受或内部处理标准 attention layout；
3. cat/contiguous 即便存在，也通常不会被 graph-cut boundary 放大成 57 个独立 segment 的大桶；
4. 编排上不会把 attention 前后拆成大量必须 cache / restore / materialize 的小 graph。
```

所以，和正常 SP 对比时，当前最异常的不是“通信慢”，而是：

```text
为了让 ggml graph-cut segment 之间继续运行，
我们把 SP layout 变换显式 materialize 了太多次。
```

### 11.2 当前实际账

当前可用基线是 q/k seq-major 后的 2GPU SP：

```text
single GPU plain total ~= 35.50s / 50 steps
2GPU SP q/k seq-major total ~= 30.34~30.42s / 50 steps

transformer elapsed avg_skip1 ~= 582.80ms
graph total avg_skip1         ~= 568.69ms
graph compute avg_skip1       ~= 485.57ms
graph comm avg_skip1          ~= 35.57ms
graph overhead avg_skip1      ~= 47.55ms
```

这说明：

```text
comm ~= 35.6ms，不是主瓶颈；
graph overhead ~= 47.6ms，也不是当前最大主矛盾；
compute ~= 485.6ms，才是把加速比压住的大桶。
```

如果只看当前 1-step compute/materialize profile，最大的 compute buckets 是：

| bucket | segments | compute | comm | layout ops | materialize | 主要异常 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sp.single.attn_head_to_seq` | 38 | **161ms** | 6ms | 1710 | **9044MiB** | attention 前后 `CONT` |
| `flux.single_blocks.tail` | 38 | **95ms** | 0ms | 503 | **9869.84MiB** | concat + qkv_mlp view materialize |
| `sp.double.attn_head_to_seq` | 19 | **92ms** | 9ms | 1330 | **7443.25MiB** | double txt/img qkv concat + `CONT` |
| `sp.single.qkv_seq_to_head` | 38 | **80ms** | 16ms | 997 | **13745.84MiB** | qkv input view + send combine |
| `sp.double.qkv_seq_to_head` | 38 | **47ms** | intrusive profile outlier | 999 | **6813.17MiB** | qkv input view + send combine |
| `flux.double_blocks.tail` | 19 | **44ms** | 0ms | 543 | **999.17MiB** | smaller tail restore |

这里最重要的是：

```text
sp.single.attn_head_to_seq + sp.double.attn_head_to_seq
  compute = 161 + 92 = 253ms

253 / 485.57 ~= 52%
```

也就是说：

```text
当前 graph compute 约一半消耗在 attention 前后 layout-heavy segment。
```

这就是和正常 SP 相比最不正常的地方。

### 11.3 为什么 `attn_head_to_seq` 这个名字会误导

`sp.*attn_head_to_seq` 不是单纯的 all-to-all helper。

它实际混着：

```text
qkv recv output restore
q/k/v 进入 RoPE / attention 前的 layout conversion
double stream txt/img q/k/v concat
attention fallback 对 v 的 permute -> cont
attention output split 成 head layout
head_to_seq recv placeholder
部分 RMSNorm / RoPE elementwise / attention core op
```

所以不能再说：

```text
head_to_seq 通信慢
```

应该说：

```text
attention 前后 layout bridge 慢。
```

### 11.4 `sp.single.attn_head_to_seq` 的异常细节

当前 q/k seq-major 后，`sp.single.attn_head_to_seq` 仍然是最大 bucket：

```text
compute             = 161ms
comm                = 6ms
materialize_bytes   = 9044MiB
CONT                = 8075MiB
permute_view_to_cont= 6137MiB
```

stage breakdown：

| stage | bytes | 解释 |
| --- | ---: | --- |
| `attention.qkv_layout` | **2907MiB** | q/k/v 进入 attention 前被重新 layout |
| `qkv.recv_output_restore` | **1938MiB** | q/k 从 seq_to_head recv_flat 恢复输出 layout |
| `permute_to_cont_layout` | **1292MiB** | 通用 `PERMUTE(view)->CONT` |
| `attention.output_split_head` | **969MiB** | attention 输出拆回 head_to_seq 输入 |
| `attention.qkv_from_boundary` | **969MiB** | boundary input 再 materialize |
| `comm.recv_placeholder` | **969MiB** | comm placeholder output |

这里的异常不是 RoPE 数学，也不是 attention core，而是：

```text
recv restore -> attention/RoPE layout -> v layout -> output split
```

这条链每个 single block 都在重复。

### 11.5 `sp.double.attn_head_to_seq` 的异常细节

`sp.double.attn_head_to_seq` 也很明显：

```text
compute             = 92ms
comm                = 9ms
materialize_bytes   = 7443.25MiB
CONT                = 4522MiB
CONCAT              = 2436.75MiB
permute_view_to_cont= 3068.50MiB
```

stage breakdown：

| stage | bytes | 解释 |
| --- | ---: | --- |
| `double.attn_qkv_concat` | **2907MiB** | txt/img q/k/v concat 后又进入 attention layout |
| `qkv.recv_output_restore` | **1453.5MiB** | img/txt qkv_seq_to_head recv output restore |
| `concat_layout` | **983.25MiB** | concat 链本身 |
| `attention.output_split_head` | **969MiB** | attention 输出拆 txt/img head |
| `permute_to_cont_layout` | **646MiB** | 额外 permute/view materialize |
| `comm.recv_placeholder` | **484.5MiB** | comm placeholder output |

这个和正常 Torch SP 的差异更直观：

```text
Torch 通常会把 txt/img concat 视为少量 cat/layout work；
当前 ggml graph-cut 路径里，concat 后又被 attention layout 固化，
再在 output split 时继续 materialize。
```

### 11.6 第二异常：single tail concat/materialize

`flux.single_blocks.tail` 也不正常，但它不是当前第一刀：

```text
compute           = 95ms
materialize_bytes = 9869.84MiB
CONCAT            = 4947MiB
CONT              = 4871.84MiB
```

stage breakdown：

```text
concat_layout              ~= 4972.5MiB
qkv.input_view_materialize ~= 3876MiB
head_to_seq.recv_restore   ~= 969MiB
```

它和 Torch 对比也偏异常：Torch 的 `cat + linear` 通常不会在 graph-cut segment 里变成这么大的显式 layout bucket。

但已经验证过：

```text
split-linear 能降 materialize，但会改变 floating-point accumulation order；
Flux hash 不一致；
compute 还从 98ms 变成 110ms。
```

所以这个 bucket 是真问题，但不能用 split-linear 修。

### 11.7 第三异常：qkv_seq_to_head send/input materialize

`qkv_seq_to_head` 也是 layout-heavy：

```text
sp.single.qkv_seq_to_head:
  compute           = 80ms
  materialize_bytes = 13745.84MiB
  qkv.send_combine  = 5814MiB
  qkv.input_view_materialize = 4845MiB

sp.double.qkv_seq_to_head:
  compute           = 47ms
  materialize_bytes = 6813.17MiB
  qkv.send_combine  = 2907MiB
  qkv.input_view_materialize = 2448MiB
```

这说明 helper send buffer 构造也不是正常 SP 该有的理想状态。

但已经验证过一个重要 negative result：

```text
直接跳过 q/k/v input cont 会导致 send_flat 顺序错误；
packed-qkv one-materialize helper 虽然 CPU oracle 过了，但真实 2GPU 变慢到 35.82s。
```

所以 qkv path 是问题，但不是下一步可以靠“再猜一个 pack”解决的问题。它必须以旧 send_flat 为 oracle，并且要证明真实 CUDA profile 变快。

### 11.8 当前真正的瓶颈排名

按“和正常 SP 相比最不正常 + 对加速比影响最大”排序：

```text
P0. attention 前后 layout bridge
    bucket:
      sp.single.attn_head_to_seq
      sp.double.attn_head_to_seq
    abnormal ops:
      CONT, CONCAT, PERMUTE(view)->CONT
    why:
      253ms compute，占 graph compute 约 52%；
      comm 只有约 15ms；
      materialize 超过 16GiB。

P1. single tail concat/materialize
    bucket:
      flux.single_blocks.tail
    abnormal ops:
      CONCAT, CONT
    why:
      95ms compute；
      materialize 约 9.9GiB；
      但 split-linear 不可用，因为 hash 不一致。

P2. qkv_seq_to_head send/input materialize
    bucket:
      sp.single.qkv_seq_to_head
      sp.double.qkv_seq_to_head
    abnormal ops:
      q/k/v view materialize, send_combine concat/cont
    why:
      compute 合计约 127ms；
      materialize 很大；
      但已经证明简单 skip cont / packed one-materialize 不是正确优化。

P3. runner overhead / runtime input / collect_future / pe cache
    why:
      已经从 70ms 级降到 47ms 级；
      现在不是最主要矛盾。

P4. NCCL/all-to-all
    why:
      avg comm 约 35ms；
      即使降到 0 也不足以解释 1.2x 的低加速比。
```

### 11.9 下一步应该怎么推进

下一步不要再泛泛地说“优化 SP helper”，而要验证这几个具体 layout contract：

```text
1. v attention layout
   当前 v 是 [head_dim, shard_heads, full_seq, batch]；
   ggml_ext_attention_ext fallback 立刻做:
     v -> permute -> cont
   这在 single/double 每个 attention block 都出现。

2. attention output split layout
   attention 输出 reshape 成 head layout 后，
   double stream 还要 split txt/img 并 ggml_cont；
   single stream 也有 output_split_head 969MiB。

3. double txt/img q/k/v concat layout
   q/k/v concat 之后又进入 attention qkv layout；
   double.attn_qkv_concat 单独 2907MiB。

4. qkv recv output restore
   q/k/v 从 seq_to_head recv_flat 恢复后，马上被下游再次换 layout；
   这是典型 graph-cut boundary materialize tax。
```

对应的实验必须满足：

```text
默认路径不变；
env gated；
Flux hash 必须一致；
CPU helper oracle 必须通过；
先看 materialize stage 是否下降，再看 50-step；
不碰 third_party/ggml；
不改 collective 接口；
不改数学 kernel。
```

当前最清晰的一刀不是 packed qkv，而是：

```text
验证 v attention layout contract：
让 seq_to_head 的 v 输出直接匹配 ggml_ext_attention_ext fallback 所需 layout，
或者在 Flux SP attention 里使用一个明确的 v-prepared branch，
目标是消掉每个 block 中 v_attn 的 PERMUTE(view)->CONT。
```

如果这刀只移动归因、不降总 compute，就撤回。
如果它能稳定降低：

```text
sp.single.attn_head_to_seq attention.qkv_layout
sp.double.attn_head_to_seq double.attn_qkv_concat / attention.qkv_layout
permute_view_to_cont
```

再继续处理 attention output split 和 double txt/img concat。

## 12. SP attention backend A/B: 一个明确的 Torch/正常 SP 差异点

### 12.1 结论

本轮在 GPU `1,3` 上确认了一个非常明确的异常：

```text
非 SP Flux / Rope::attention:
  ggml_ext_attention_ext(..., skip_reshape=true, flash_attn=ctx->flash_attn_enabled)

Flux SP custom flux_sp_attention 旧路径:
  ggml_ext_attention_ext(..., skip_reshape=true, flash_attn=false)
```

也就是说，Flux SP 自定义 attention 之前写死使用 fallback attention。正常 Torch/Ulysses SP 通常会走 fused/scaled-dot-product attention，而当前 clean Flux SP 在 attention core 上走的是：

```text
v permute -> cont
kq = mul_mat(k, q)
softmax
out = mul_mat(v, kq)
```

这和正常 SP 后端相比是一个大差异，不是 RoPE、NCCL 或小 reshape 问题。

### 12.2 1-step graph-cut A/B

命令环境共同部分：

```bash
ED_RUNTIME_CONST_CACHE=1
ED_FLUX_SP_QK_SEQ_MAJOR=1
ED_PROFILE_GRAPH_CUTS=1
ED_PROFILE_GRAPH_CUTS_TOP=0
ED_PROFILE_GRAPH_CUTS_COMPUTE_TOP=8
ED_PROFILE_GRAPH_CUTS_MATERIALIZE_TOP=8
--devices 1,3
--sp-size 2
--steps 1
```

结果：

| mode | graph compute | single attn_head_to_seq compute | double attn_head_to_seq compute | attention op |
| --- | ---: | ---: | ---: | --- |
| clean fallback | 526ms | 156ms | 92ms | `MUL_MAT` fallback |
| `ED_FLUX_SP_FLASH_ATTN=1` | 433ms | 106ms | 59ms | `FLASH_ATTN_EXT` |

差值：

```text
graph compute:              526ms -> 433ms, 省约 93ms
sp.single.attn_head_to_seq:  156ms -> 106ms, 省约 50ms
sp.double.attn_head_to_seq:   92ms ->  59ms, 省约 33ms
```

这说明之前归因到 `attn_head_to_seq` 的大桶里，确实有相当一部分是 fallback attention core 和它周围的 layout/materialize，而不是 all-to-all 通信。

### 12.3 50-step real run A/B

无 graph-cut profile 的真实 50-step 结果，先看双卡 SP 自身的 fallback vs flash 差异：

| mode | 50-step sampling time | output hash |
| --- | ---: | --- |
| clean fallback | 28.70s ~ 28.71s | `9710594fc677b7762ca44c384051307c247d3e6fe2983ed85cd357a8c5fcbbe4` |
| `ED_FLUX_SP_FLASH_ATTN=1` | 24.86s ~ 24.87s | `f44e38f569e3a17d0f5c397095f6d736e3ed170a28397173bc5c5ee49e6c7bdc` |

实际收益：

```text
28.70s -> 24.86s
省约 3.84s / 50 steps
约 13.4% faster
```

但不能用 fallback 单卡直接除以 flash 双卡来宣称 SP 已经达到 1.4x，因为那不是同后端公平对比。本轮也补跑了单卡：

| mode | 50-step sampling time | output hash |
| --- | ---: | --- |
| 1GPU fallback | 35.42s | `9710594fc677b7762ca44c384051307c247d3e6fe2983ed85cd357a8c5fcbbe4` |
| 2GPU SP fallback | 28.70s ~ 28.71s | `9710594fc677b7762ca44c384051307c247d3e6fe2983ed85cd357a8c5fcbbe4` |
| 1GPU diffusion flash | 29.56s | `d3c1bdb9ff94d84853f59ac56ce140664ebc3c4577d2bdd1438f577fc744dad5` |
| 2GPU SP diffusion flash | 24.86s ~ 24.87s | `f44e38f569e3a17d0f5c397095f6d736e3ed170a28397173bc5c5ee49e6c7bdc` |

所以公平 speedup 是：

```text
fallback backend:
  35.42 / 28.70 ≈ 1.23x

flash backend:
  29.56 / 24.86 ≈ 1.19x
```

这个校正很重要：flash attention 说明当前 clean SP fallback attention 后端确实明显慢，但它并没有单独解决双卡加速比偏低的问题。同后端对比下，SP 仍然只有约 1.2x，剩余主矛盾还是 graph-cut/layout/materialize/segment 结构。

1-step image hash 也不同：

```text
clean fallback 1-step hash:
  60b7568181785a32da06748fe2ce718dee33b0032940c0bc49a37025b8235fc9

flash attention 1-step hash:
  0a67571effefacd304ae0cdc15f4563f7e5f8473ac415d855f8a9930cc1adf78
```

因此这条路径不是 bitwise-clean 默认优化；它只能作为：

```text
1. root-cause 证据：Flux SP fallback attention 明显慢于 fused attention；
2. 显式非 bitwise 性能选项：用户接受 hash 不同才打开；
3. clean 优化方向指示：如果必须保持 hash，就要优化 fallback attention 周围的 layout/materialize tax。
```

### 12.4 当前瓶颈排序更新

加入 attention backend A/B 后，当前排序应该更新为：

```text
P0. Flux SP attention 后端不一致
    clean 路径仍走 fallback attention；
    flash A/B 证明可省约 93ms graph compute / 3.84s per 50 steps；
    但 hash 不一致，不能默认打开；
    且同后端 flash speedup 仍只有约 1.19x，所以它不是最终答案。

P1. clean fallback attention 周围 layout/materialize
    仍然有 v_attn PERMUTE(view)->CONT、q/k/v concat、qkv recv restore；
    这是必须保持 hash 时最该继续打的部分。

P2. flux.single_blocks.tail
    约 97ms compute；
    split-linear 已证明 hash 不一致，不能走改变 matmul 累加顺序的路线。

P3. qkv_seq_to_head send/input materialize
    仍然很重；
    但 packed-qkv pack 实验真实变慢，不能再盲目推进。

P4. runner overhead / pe cache / collect_future
    已经被修到次要位置。

P5. NCCL/all-to-all
    不是主瓶颈。
```

### 12.5 下一步判断

如果目标允许非 bitwise 输出：

```text
把 Flux SP attention 接入已有 flash-attention 配置路径，作为显式性能选项。
本轮已通过 --diffusion-flash-attention 接通：
  CLI flag 1-step hash == ED_FLUX_SP_FLASH_ATTN=1 诊断 hash
```

如果目标必须 bitwise/hash 不变：

```text
不能默认打开 flash attention；
下一步应该优化 clean fallback 路径里的 layout/materialize：
  v_attn PERMUTE(view)->CONT
  double q/k/v txt+img CONCAT -> PERMUTE(view)->CONT
  qkv recv_output_restore
  attention output split/head_to_seq restore
```

这个 A/B 把问题从“SP helper 泛泛很慢”进一步压到了：

```text
当前 Flux SP attention core/back-end 和正常 SP 后端不一致；
在不能接受 hash 变化时，clean 路径必须降低 fallback attention 前后的 layout materialization tax。
```

## 13. Batched Collective 拆分实验：否定结论

本轮又做了两个只用于诊断的默认关闭实验，目标是验证：

```text
当前 qkv/head_to_seq batched helper 的合包 concat/materialize 是否比多发几次 all-to-all 更贵？
```

结论很明确：

```text
不是。
在当前 ggml graph-cut runner 下，把 batched collective 拆成更多独立 collective 会显著增加 segment / comm op / graph overhead，并且总 compute 也会变高。
```

### 13.1 double head_to_seq 拆成 txt/img 两次 all-to-all

实验：

```text
ED_FLUX_SP_SPLIT_DOUBLE_HEAD_TO_SEQ=1
```

语义结果：

```text
1-step flash hash 与 baseline flash 完全一致：
0a67571effefacd304ae0cdc15f4563f7e5f8473ac415d855f8a9930cc1adf78
```

但 profile 明显变差：

| metric | baseline batched | split txt/img |
| --- | ---: | ---: |
| graph segments | 193 | 212 |
| comm ops | 135 | 154 |
| graph compute | 435ms | 492ms |
| transformer 1-step | 1072ms | 1090ms |
| `sp.double.attn_head_to_seq` compute | 59ms | 113ms |
| `sp.double.attn_head_to_seq` materialize | 6474MiB | 10498MiB |

所以这条路不能继续推。它说明：

```text
double head_to_seq 的 batched 合包虽然有 concat/cont tax，
但拆成两个 all-to-all 会让同一段 attention/qkv restore work 被复制到更多 graph segment 里，
整体更慢。
```

### 13.2 qkv_seq_to_head 拆成 q/k/v 三次 all-to-all

第一次直接拆会触发：

```text
GGML_ASSERT(ggml_is_contiguous(a)) failed in ggml_reshape_4d()
```

原因是单 tensor helper 不像 batched helper 一样先对 packed q/k/v view 做 `ggml_cont`。这再次证明：

```text
q/k/v input cont 不是随便能删的优化；
它承担了 packed qkv view -> helper canonical layout 的语义。
```

补上 q/k/v input cont 后，实验可以运行，且 hash 与 baseline flash 一致：

```text
0a67571effefacd304ae0cdc15f4563f7e5f8473ac415d855f8a9930cc1adf78
```

但性能更差：

| metric | baseline batched | split q/k/v |
| --- | ---: | ---: |
| graph segments | 193 | 345 |
| comm ops | 135 | 287 |
| graph compute | 435ms | 527ms |
| graph total | 1054ms | 1268ms |
| transformer 1-step | 1072ms | 1292ms |
| comm-name summary | `qkv_seq_to_head=76` | `q/k/v_seq_to_head=76/76/76` |

同时 split 后出现新的大桶：

```text
sp.single.other:
  segments=114
  compute=150ms
  materialize=9260MiB

sp.double.other:
  segments=114
  compute=79ms
  materialize=4451MiB
```

这说明拆 q/k/v 后，原先一个 qkv segment 里的工作被拆散成更多 graph-cut segment，并且大量 q/k/v output restore、norm、attention 前 layout 被重新归入 `other` bucket。它不是优化方向。

### 13.3 更新后的判断

这两个实验把方向进一步收窄：

```text
1. 不要通过拆 batched collective 来减少 concat/materialize。
   当前 graph-cut runner 对 segment 数和 comm op 数很敏感；
   拆 collective 会扩大 graph fragmentation。

2. qkv/head_to_seq 的 batched 合包是当前实现里较合理的折中。
   问题不是“batched helper 不该合包”，而是合包前后仍有太多显式 layout restore。

3. 下一步真正要打的是减少单个 segment 内的 materialize，
   不是把一个 segment 拆成多个 segment。
```

因此，当前最可信的主瓶颈仍然是：

```text
P0. `flux.single_blocks.tail`
    `single.tail.attn_mlp` concat 约 4845MiB；
    `qkv.input_view_materialize` 约 3876MiB；
    约 94~97ms compute。

P1. `sp.single.attn_head_to_seq`
    约 101~107ms compute；
    主要是 attention q/k/v layout、recv restore、permute->cont。

P2. `sp.double.attn_head_to_seq`
    约 57~59ms compute；
    batched helper 不应拆，但 q/k/v txt+img concat 和 recv/layout restore 仍重。

P3. `qkv_seq_to_head`
    batched helper 不应拆；
    后续只能做 bitwise/layout-oracle 等价的 fewer-materialize，而不是增加 collective 数。
```

和 Torch/Ulysses 后端相比，当前 ggml SP 的明显异常不是 collective 算子本身，而是：

```text
每个 block 周围显式构造大量 contiguous layout buffer；
graph-cut boundary 让 q/k/v、attention output、single tail concat 被反复 materialize；
拆更多 graph segment 只会让问题更糟。
```

## 14. 2026-06-08 更新：当前有效路径与新的失败实验

这一轮重新把边界校准了一遍：有效优化只保留已经在 50-step 路径上证明加速、且 hash 不变的项；只让 materialize bytes 下降但总时间变慢的实验要撤回。

### 14.1 当前有效最快路径

同后端 flash 对比，当前最好的双卡 SP 路径是：

```text
ED_RUNTIME_CONST_CACHE=1
ED_FLUX_SP_QK_SEQ_MAJOR=1
ED_FLUX_SP_FLASH_V_SEQ_MAJOR=1
ED_FLUX_SP_SPLIT_SINGLE_LINEAR1=1
--diffusion-flash-attention
--devices 1,3
--sp-size 2
```

50-step 结果：

```text
2GPU SP flash baseline:
  /tmp/flux_2gpu_sp_flash_current_50.log
  24.85s ~ 24.86s

2GPU SP flash + V seq-major:
  /tmp/flux_2gpu_sp_flash_v_seq_major_50.log
  24.27s ~ 24.28s

2GPU SP flash + V seq-major + split single linear1:
  /tmp/flux_2gpu_sp_flash_v_seq_major_split_single_linear1_50.log
  22.12s ~ 22.13s
```

同后端 1GPU flash 基线：

```text
1GPU diffusion flash:
  29.56s

当前最好 2GPU SP flash:
  22.12s

speedup:
  29.56 / 22.12 ~= 1.34x
```

hash 验证：

```text
flash 1-step hash:
  0a67571effefacd304ae0cdc15f4563f7e5f8473ac415d855f8a9930cc1adf78

flash 50-step hash:
  f44e38f569e3a17d0f5c397095f6d736e3ed170a28397173bc5c5ee49e6c7bdc
```

`ED_FLUX_SP_SPLIT_SINGLE_LINEAR1=1` 的作用不是拆 `linear2`，也不是改变 `linear2(concat(attn, mlp))` 的累加顺序。它只把 single block 的 `linear1` 输出列切成：

```text
qkv_mlp = linear1[:, 0 : hidden_size * 3]
mlp     = linear1[:, hidden_size * 3 : end]
```

因此仍然保留：

```text
attn_mlp = concat(attn_flat, mlp)
output   = linear2(attn_mlp)
```

这个路径的 1-step profile 对比 V seq-major-only：

```text
graph compute:
  433ms -> 383ms

flux.single_blocks.tail compute:
  96ms -> 69ms

flux.single_blocks.tail materialize:
  9869.84MiB -> 5993.84MiB

single tail qkv.input_view_materialize:
  3876MiB -> 0
```

这是当前最强、且可保留的有效优化。

### 14.2 已测试并撤回：split attention qkv projection

实验：

```text
ED_FLUX_SP_SPLIT_ATTN_QKV_PROJ=1
```

做法是仿照 single `linear1` 的成功路径，在 `SelfAttention::pre_attention_sp()` 里把 qkv projection 从：

```text
qkv = qkv_proj(x)
q/k/v = view(qkv)
batched qkv_seq_to_head(q, k, v)
```

改成：

```text
q = qkv_proj output slice 0
k = qkv_proj output slice 1
v = qkv_proj output slice 2
batched qkv_seq_to_head(q, k, v)
```

这个实验没有增加 collective 数，也没有拆 batched all-to-all，但 1-step profile 已经显著变慢：

```text
baseline current best profile:
  /tmp/flux_2gpu_sp_flash_v_seq_major_split_single_linear1_profile_1.log
  flux sampling completed ~= 0.96s
  graph compute           ~= 383ms

split attention qkv projection:
  /tmp/flux_2gpu_sp_flash_split_attn_qkv_proj_profile_1.log
  flux sampling completed ~= 3.52s
  graph compute           ~= 988ms
```

局部看似命中了一个 materialize bucket：

```text
sp.double.qkv_seq_to_head qkv.input_view_materialize:
  2448MiB -> 25.5MiB
```

但代价更大：

```text
sp.double.qkv_seq_to_head compute:
  55ms -> 147ms

sp.single.qkv_seq_to_head compute:
  63ms -> 153ms

sp.single.attn_head_to_seq compute:
  102ms -> 285ms

flux.single_blocks.tail compute:
  69ms -> 163ms
```

结论：

```text
只让 materialize bytes 下降不等于优化成功。
把一个大 qkv projection 拆成多个 projection/slice graph，会增加 matmul 和 graph scheduling 压力；
在当前 ggml graph-cut runner 下，这条路明显变慢，实验代码已撤回。
```

### 14.3 当前剩余主瓶颈

当前最好路径下，主要瓶颈仍是 ggml graph-cut/layout/materialize tax，而不是通信本身：

```text
sp.single.attn_head_to_seq:
  compute ~= 102ms
  materialize ~= 7106MiB
  主要是 qkv.recv_output_restore、attention q/k layout、qkv_from_boundary、recv placeholder restore。

flux.single_blocks.tail:
  compute ~= 69ms
  materialize ~= 5994MiB
  single.tail.attn_mlp concat ~= 4845MiB。

sp.single.qkv_seq_to_head:
  compute ~= 63ms
  materialize ~= 13746MiB
  qkv.send_combine ~= 5814MiB
  qkv.input_view_materialize ~= 4845MiB。

sp.double.attn_head_to_seq:
  compute ~= 56ms
  materialize ~= 5990MiB。

sp.double.qkv_seq_to_head:
  compute ~= 55ms
  materialize ~= 6813MiB。
```

和 Torch/Ulysses 相比，异常点不是 SP 理论，也不是 collective 数学语义，而是：

```text
1. q/k/v 和 attention output 在 graph-cut boundary 周围反复 reshape/permute/cont；
2. single tail 的 concat + linear2 在 ggml graph-cut 路径下变成显式大 materialize；
3. qkv_seq_to_head 的 batched send_combine 仍有大 cont+concat；
4. 任何增加 collective 数或大 GEMM 次数的实验都会快速变慢。
```

下一步应只考虑同时满足这些条件的优化：

```text
不增加 collective 数；
不增加 graph segment 数；
不拆大 GEMM 成更多 GEMM；
不改变 linear2(concat(...)) 的浮点累加顺序；
不碰 third_party/ggml；
hash 必须保持当前 flash backend 一致。
```

### 14.4 无效性能样本：GPU 1/3 被外部任务占用

2026-06-08 晚上曾重跑当前最好 50-step 路径：

```text
/tmp/flux_2gpu_sp_flash_v_seq_major_split_single_linear1_50_confirm_20260608.log
sampling completed = 137.03s
```

这个结果不能作为性能回归证据。运行期间 `nvidia-smi` 显示 GPU 1 和 GPU 3 并不空闲，而是被 Step1X-Edit Python 评测进程占用：

```text
pid 820787: GPU 1, Step1X-Edit shard_id=1
pid 820791: GPU 3, Step1X-Edit shard_id=3
```

同时 GPU 0/2/5 也有同一组评测进程高负载运行。该样本是外部 GPU 竞争下的慢跑，不能和空卡时的：

```text
/tmp/flux_2gpu_sp_flash_v_seq_major_split_single_linear1_50.log
22.12s ~ 22.13s
```

直接比较。后续性能确认必须先保证参与 SP 的两张卡实际空闲。
