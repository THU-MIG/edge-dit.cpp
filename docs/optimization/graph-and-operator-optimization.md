# Graph and Operator Optimization

[← Back to performance](../performance.md) | [← Back to README](../../README.md)

## 1. Overview

Diffusion Transformer inference is dominated not only by matrix multiplication
and attention, but also by repeated normalization, adaptive modulation,
rotary position encoding, tensor-layout conversion, and intermediate tensor
materialization.

When these patterns are expressed only through generic graph operators, they
may introduce:

- additional kernel launches;
- repeated reads and writes to device memory;
- temporary tensors and contiguous copies;
- layout and dtype conversions around attention;
- large graph workspaces and high peak VRAM usage.

edge-dit.cpp addresses these costs at two levels:

1. **Operator-level specialization**

   Frequently repeated DiT computation patterns are represented as semantic
   operators and mapped to specialized backend implementations.

2. **Graph-level execution optimization**

   Tensor layouts and materialization are optimized across operator
   boundaries, while large graphs can be divided into memory-bounded execution
   segments.

The optimized paths preserve the original model computation. Unsupported
tensor shapes, layouts, dtypes, or backend configurations fall back to the
generic ggml execution path.

Approximate computation reuse is documented separately in
[Computation reuse](computation-reuse.md), while distributed execution is
covered in [Parallel execution](parallel-execution.md).

---

## 2. Operator-level Optimization

### 2.1 DiT-specific fused operators

DiT blocks repeatedly apply combinations of normalization, conditioning,
position encoding, and gated residual updates. Expressing each step as a
separate graph operator can create unnecessary intermediate tensors, kernel
launches, and device-memory traffic.

edge-dit.cpp provides specialized execution paths for common DiT patterns,
including:

- fused RMS normalization, affine scaling, and optional dtype conversion;
- adaptive modulation of the form

  ```text
  y = x * (1 + scale) + shift
  ```

- gated residual updates of the form

  ```text
  y = residual + gate * x
  ```

- layout-aware rotary position encoding for image and video token layouts.

These operators preserve high-level DiT semantics while allowing the backend
to combine multiple low-level operations into fewer execution stages.

| Primitive | Combined operations | Main benefit |
|---|---|---|
| RMS normalization | Reduction, normalization, affine scaling, and optional cast | Fewer temporary tensors and kernel launches |
| Adaptive modulation | Scale, identity addition, shift, and broadcasting | Fewer element-wise memory passes |
| Gated residual | Gate multiplication and residual addition | Avoids intermediate residual tensors |
| Rotary position encoding | Position rotation and layout-aware output | Reduces transpose and contiguous conversion |

The model implementation remains backend-independent. Before selecting an
optimized path, the runtime checks properties such as:

- tensor shape and rank;
- input and output dtype;
- tensor stride and contiguity;
- broadcast compatibility;
- backend capability.

When the requirements of an optimized implementation are not satisfied,
execution falls back to the generic path without changing model semantics.

### 2.2 Attention execution optimization

Attention performance depends on both the SDPA kernel and the preparation of
Q, K, and V tensors.

A generic DiT attention path may contain several layout and conversion stages:

```text
projection
→ reshape
→ RoPE
→ concatenate
→ permute
→ contiguous conversion
→ dtype conversion
→ SDPA
```

For long image and video token sequences, these operations can generate
substantial memory traffic even though their arithmetic cost is relatively
small.

edge-dit.cpp optimizes the complete attention execution path through:

- fused Q, K, and V tensor packing;
- combined layout and dtype conversion;
- direct generation of backend-compatible tensor layouts;
- shape-, stride-, and dtype-aware cuDNN SDPA dispatch;
- cached cuDNN execution plans;
- optional execution-plan prewarming;
- fallback to the generic attention path for unsupported configurations.

The optimized dataflow can therefore be reduced to:

```text
projection
→ layout-aware RoPE
→ fused QKV preparation
→ SDPA
```

cuDNN SDPA plans are selected using execution properties including:

- device;
- tensor dtype;
- batch size;
- number of attention heads;
- query and key sequence lengths;
- head dimension;
- attention scale;
- padding configuration;
- Q, K, V, and output strides.

Including stride and layout information in plan selection prevents an
execution plan built for one tensor representation from being reused
incorrectly for another.

Plan construction and steady-state execution are treated separately.
Prewarming can move one-time plan construction outside the measured inference
path, while cached plans are reused across repeated denoising steps.

#### Quantized attention (SageAttention, optional)

As an opt-in alternative to the F16 SDPA path, edge-dit.cpp includes a
SageAttention-style fused kernel that runs the Q·Kᵀ score matmul in INT8
(per-block quantized, with per-channel key smoothing) while keeping the P·V
accumulation in F16. It is enabled at runtime with `ED_SAGE_ATTN=1` and is
gated at build time by `ED_ENABLE_CUDA_SAGE_ATTN` (CUDA only).

Scope and measured effect (RTX 4090, vs the default cuDNN SDPA baseline,
quality verified loss-free):

- **SD3** (head_dim 64): self-attention runs on SageAttention; DiT sampling
  is about **+5–6%** faster.
- **Wan** (video, head_dim 128): long self-attention sequences run on
  SageAttention loss-free; end-to-end gain is small (**~1%** on the 1.3B model)
  because the cuDNN SDPA baseline is already well optimized for long sequences.
- FLUX / Qwen-Image are **not** covered by this path (their head_dim-128
  attention hits a separate accuracy issue, and Qwen routes through a different
  attention entry); they continue to use the standard F16 SDPA path.

The gain is modest because attention is only part of DiT cost and the cuDNN
SDPA baseline is strong; SageAttention is provided as an optional speedup, not
a default. Unsupported shapes fall back to the standard attention path
automatically.


---

## 3. Graph-level Execution Optimization

### 3.1 Materialization and tensor-layout optimization

DiT graphs contain many operations whose arithmetic cost is small but whose
memory cost can be significant.

Typical examples include:

```text
reshape
view
permute
transpose
concatenate
contiguous
copy
dtype conversion
```

These operations do not have the same execution cost.

A `view` or `reshape` may only modify tensor metadata, while `contiguous`,
`copy`, and `concatenate` normally require actual device-memory movement and
may allocate new tensors.

edge-dit.cpp therefore distinguishes among:

- **mathematical operators**, which perform model computation;
- **metadata-only layout operators**, which reinterpret tensor storage;
- **materialization operators**, which create or copy tensor data.

The graph execution path reduces materialization through:

- replacing repeated graph patterns with fused semantic operators;
- writing outputs directly in the layout required by downstream operators;
- combining concatenation, layout conversion, and dtype conversion;
- avoiding unnecessary contiguous copies;
- allowing specialized kernels to consume supported non-default strides;
- profiling mathematical, layout, and materialization costs separately.

A key design principle is:

> Low arithmetic intensity does not imply low execution cost.

A sequence of layout and copy operators may execute fewer floating-point
operations than a matrix multiplication while still consuming considerable
time through memory bandwidth, allocation, and kernel-launch overhead.

For this reason, graph optimization in edge-dit.cpp targets both computational
operators and the data movement between them.

### 3.2 Graph-cut segmented execution

A complete DiT graph may require a large temporary workspace and retain
multiple intermediate tensors at the same time.

This is especially important for:

- high-resolution image generation;
- long video token sequences;
- large text and image conditioning inputs;
- multi-branch image-editing pipelines;
- devices with limited available VRAM.

edge-dit.cpp can divide a large graph into independently executable segments at
explicit graph-cut boundaries.

```text
Complete graph
    │
    ├── Segment 0
    │       ↓ retained boundary tensors
    ├── Segment 1
    │       ↓ retained boundary tensors
    └── Segment N
```

For each segment, the runtime tracks:

- operators assigned to the segment;
- tensors produced by previous segments;
- tensors required by later segments;
- temporary compute-buffer requirements;
- retained inputs, outputs, and boundary tensors.

This graph-cut mechanism exists in the current runtime. It supports segmented
execution, retained boundary tensors, cached graph-cut plans, communication
boundaries, and graph-cut profiling.

The approximate memory requirement of one segment can be represented as:

```text
segment memory ~= workspace + inputs + outputs + retained boundary tensors
```

Instead of executing the complete graph with one large workspace, the runtime
executes segments sequentially:

```text
execute segment
→ preserve required boundary tensors
→ release temporary workspace
→ execute next segment
```

The peak-memory requirement is therefore influenced by the largest active
segment plus the tensors retained across segment boundaries.

The primary purpose of graph segmentation is **peak VRAM control**.

It can turn a configuration that would otherwise fail with an out-of-memory
error into a runnable configuration. It may also avoid more expensive
alternatives such as repeatedly transferring model components or intermediate
states between CPU and GPU.

#### VRAM-budgeted planning and segment merging

Graph segmentation introduces additional costs, including:

- extra graph launches;
- host-side scheduling;
- synchronization points;
- longer lifetimes for boundary tensors;
- possible loss of cross-segment fusion opportunities.

For this reason, a larger number of segments is not necessarily better.

The codebase contains graph-cut plan caching and VRAM-budgeted segment merging
machinery, but the public interface, validation status, and recommended user
workflow are not yet frozen. Treat this area as an internal implementation
direction until the release validation work is complete.

TODO:

- Document the stable public entry point for VRAM-budgeted graph-cut planning.
- Document how `--max-vram` interacts with graph segmentation.
- Validate segment merging across supported model families.
- Add failure-mode documentation for models that cannot be segmented safely.
- Add benchmark methodology for memory savings versus launch/synchronization
  overhead.

The intended direction is to merge adjacent segments when the available memory
budget permits, reducing execution overhead while still respecting memory
constraints:

```text
fine-grained base segments
        ↓ memory analysis
budget-compatible segment merging
        ↓
coarser execution plan
```

The design goal is to balance two competing objectives:

- smaller segments reduce active workspace requirements;
- larger segments reduce scheduling and synchronization overhead.

#### Communication-aware execution boundaries

The same segment boundaries can expose locations where collective
communication is required during multi-worker execution.

```text
local computation
→ graph boundary
→ all-gather / all-reduce
→ graph boundary
→ following computation
```

Graph cuts do not reduce the mathematical workload of the model. They provide
the runtime structure required to coordinate memory management,
communication, and execution profiling.

Any final speedup comes from secondary benefits such as:

- avoiding CPU offload;
- keeping the workload resident on the GPU;
- distributing sufficient computation across multiple devices;
- reducing unnecessary graph fragmentation through segment merging.

If segmentation, synchronization, and communication overhead exceed these
benefits, segmented execution may be slower than full-graph execution.

Graph segmentation is therefore treated as:

> A runtime mechanism for memory-bounded and communication-aware DiT
> execution, rather than an unconditional acceleration technique.

See [Parallel execution](parallel-execution.md) for sequence-parallel layouts,
collective communication, and multi-worker execution.

---

## Related Documentation

- [Performance and optimization](../performance.md)
- [Model representation and precision](model-representation-and-precision.md)
- [Memory-efficient execution](memory-efficient-execution.md)
- [Computation reuse](computation-reuse.md)
- [Parallel execution](parallel-execution.md)
- [Build and installation](../build.md)
