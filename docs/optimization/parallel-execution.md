# Parallel Execution

[← Back to README](../../README.md)

edge-dit.cpp provides a native parallel execution path for DiT inference. The
runtime separates parallel state, collective communication, and model-specific
partitioning strategies, and integrates communication directly into ggml graph
execution.

The current implementation supports:

- pipeline-level classifier-free guidance parallelism;
- Ulysses-style sequence parallelism;
- CPU and NCCL collective communication backends;
- graph-cut and in-graph collective execution paths.

Build profiles, launch commands, and model-specific CLI options are documented
separately.

## 1. Parallel State and ggml-integrated Communication

The parallel runtime is organized into three layers:

```text
ParallelConfig
      ↓
ParallelContext
      ↓
ProcessGroup
```

`ParallelConfig` describes the process environment and requested parallel
configuration, including:

- communication backend;
- global rank and world size;
- local rank and device assignment;
- CFG and sequence-parallel degrees.

`ParallelContext` is the runtime-owned parallel state. It is created during
engine initialization and attached to `ModelRuntime`, which then exposes the
corresponding process group to DiT graph runners.

`ProcessGroup` provides a backend-independent collective interface:

- all-reduce;
- all-gather;
- all-to-all;
- broadcast;
- barrier and synchronization;
- synchronous and asynchronous execution.

CPU and NCCL implementations share this interface, so model code defines the
partitioning strategy without depending directly on a communication backend.

### Communication inside ggml execution

edge-dit.cpp supports two mechanisms for integrating communication with a
ggml graph.

#### Graph-cut communication path

In the portable path, model construction produces:

- a send-side ggml tensor;
- a receive-side placeholder tensor;
- collective metadata describing the communication operation.

The communication metadata records the collective kind, input tensor, output
tensor, reduction mode, and per-peer element count.

During graph-plan construction, the runtime:

1. maps the send and receive tensors to ggml graph nodes;
2. attaches the collective operation to the segment producing the send tensor;
3. retains the send tensor so its buffer is not recycled before communication;
4. executes the collective after the segment has completed;
5. exposes the receive tensor as an input to the following segment.

```text
ggml segment
      ↓
send tensor retained
      ↓
collective communication
      ↓
receive tensor bound
      ↓
next ggml segment
```

Communication is therefore represented by graph tensors and graph
dependencies rather than by model-specific host-side calls scattered through
individual DiT implementations.

#### In-graph CUDA collective path

For supported CUDA configurations, collective communication can be represented
directly as a ggml custom operator.

```text
packing operator
      ↓
GGML_OP_CUSTOM: all-to-all / all-gather
      ↓
receive-side operator
```

The CUDA backend recognizes the collective node and invokes the NCCL operation
asynchronously on the CUDA stream used by the ggml runner.

This preserves graph dependency ordering and allows the collective output to be
consumed by ordinary downstream ggml operators without leaving the graph
execution path.

The graph-cut path remains available as the generic fallback when an in-graph
collective is unavailable.

## 2. CFG Parallelism

### Principle

Standard classifier-free guidance evaluates two model predictions at every
denoising step:

- an unconditional prediction, `p_uncond`;
- a conditional prediction, `p_cond`.

They are combined as:

```text
p_cfg = p_uncond + guidance_scale * (p_cond - p_uncond)
```

The two model evaluations are independent until the final combination, making
them suitable for branch-level parallel execution.

### Implementation

edge-dit.cpp implements CFG parallelism at the pipeline level:

```text
Rank 0
└── unconditional DiT branch

Rank 1
└── conditional DiT branch

          ↓ all-gather

conditional and unconditional predictions
          ↓
CFG combination
          ↓
sampler update
```

The parallel state determines which conditioning branch is executed by each
rank. Both ranks use the same model parameters and denoising timestep but
receive different conditioning inputs.

After DiT execution, the branch predictions are gathered and combined before
the sampler advances to the next latent state.

CFG parallelism communicates only the final prediction tensor of each branch,
rather than intermediate activations within every Transformer block. Its
communication cost is therefore relatively small compared with sequence
parallelism.

The current implementation targets the standard two-branch CFG configuration:

```text
cfg_parallel_size = 2
world_size        = 2
```

Models that do not use a separate conditional and unconditional forward pass
do not benefit from this parallel mode.

## 3. Ulysses Sequence Parallelism

### Principle

Image and video DiTs may process thousands or tens of thousands of tokens.
Their attention computation and activation memory increase rapidly with
sequence length.

edge-dit.cpp follows the Ulysses sequence-parallel pattern. Let:

- `P` be the number of sequence-parallel ranks;
- `S` be the global sequence length;
- `H` be the number of attention heads;
- `D` be the head dimension.

At the beginning of a Transformer block, each rank owns a local sequence shard:

```text
X_rank shape: [B, S / P, H * D]
```

Token-wise operations, including normalization, modulation, linear projection,
and MLP computation, can be performed independently on this local shard.

After Q, K, and V projection, each rank holds:

```text
Q_rank, K_rank, V_rank shape: [B, S / P, H, D]
```

Full attention cannot be computed directly in this layout because every query
head must access keys and values from the complete sequence.

Ulysses uses All-to-All to exchange sequence partitions for attention-head
partitions:

```text
Before All-to-All:

each rank owns
[S / P tokens, H heads]

              ↓ sequence-to-head All-to-All

After All-to-All:

each rank owns
[S tokens, H / P heads]
```

Formally, the redistributed tensors are:

```text
Q_rank, K_rank, V_rank after redistribution:
[B, S, H / P, D]
```

Each rank now has the complete sequence for a subset of attention heads and can
compute exact full-sequence attention locally:

```text
O_rank = softmax((Q_rank * transpose(K_rank)) / sqrt(D)) * V_rank
```

A second All-to-All reverses the redistribution:

```text
attention output
[S tokens, H / P heads]

              ↓ head-to-sequence All-to-All

local output
[S / P tokens, H heads]
```

The resulting tensor can continue through the output projection, residual
connection, and MLP while remaining sequence-sharded.

The complete attention path is therefore:

```text
sequence-sharded hidden states
        ↓ local QKV projection
sequence-sharded Q/K/V
        ↓ sequence-to-head All-to-All
head-sharded full-sequence Q/K/V
        ↓ local full-sequence attention
head-sharded attention output
        ↓ head-to-sequence All-to-All
sequence-sharded attention output
```

This preserves the original full-attention semantics. It does not approximate
or restrict the attention matrix.

The main layout constraints are:

- the sequence length must be divisible by the SP degree, or padded;
- the number of attention heads must be divisible by the SP degree;
- every rank must use the same tensor shape and collective order.

### Implementation

The edge-dit.cpp SP runtime represents each redistribution explicitly.

#### Sequence state

Sequence partitioning produces an `SPSequenceSplit` state containing:

- original sequence length;
- padded sequence length;
- local sequence length;
- padding size;
- rank and world size;
- padded input tensor;
- local sequence view;
- contiguous local tensor.

```text
global tensor
      ↓ optional padding
padded global tensor
      ↓ rank-specific view
local sequence view
      ↓ optional materialization
contiguous local tensor
```

The explicit state prevents model implementations from independently
reconstructing sequence offsets, padding, and local tensor shapes.

#### Redistribution state

All-to-All redistribution is represented by an `SPAllToAll4DLayout` or
`SPAllToAll4DBatchLayout` state.

The state records:

- redistribution direction;
- send and receive tensors;
- number of global and local heads;
- global and local sequence lengths;
- head dimension and batch size;
- elements communicated to each peer;
- post-communication output layout.

Two directions are supported:

```text
kSeqToHead
sequence-sharded → head-sharded

kHeadToSeq
head-sharded → sequence-sharded
```

The send tensor, collective output, and restored tensor layout remain ordinary
ggml tensors. They can therefore participate in graph dependency analysis,
memory planning, graph-cut execution, and profiling.

### SP-specific Operator Optimization

A direct implementation of Ulysses using generic tensor operators may produce
a long chain around every collective:

```text
Q / K / V
→ concatenate
→ reshape
→ permute
→ contiguous
→ dtype conversion
→ All-to-All
→ reshape
→ split
→ permute
→ contiguous
→ RoPE
→ attention
```

For DiT inference, these layout and materialization operations can significantly
reduce the benefit obtained from distributing attention.

edge-dit.cpp therefore provides SP-specific operator paths around the
collectives.

#### Peer-major QKV packing

The sequence-to-head send buffer must be arranged by destination rank and
destination head group.

The packed path combines:

- Q, K, and V concatenation;
- head-group partitioning;
- sequence-to-head layout conversion;
- peer-major flattening;
- optional F32-to-F16 conversion.

```text
local Q, K, V
        ↓ packed QKV operator
peer-major communication buffer
        ↓ All-to-All
rank-major receive buffer
```

Packing Q, K, and V together allows the first Ulysses redistribution to use one
combined communication buffer instead of three independently materialized
tensor paths.

The runtime provides variants for:

- F32 communication;
- F16 communication;
- mixed Q/K and V representations;
- paired text and image streams used by FLUX double-stream blocks.

#### Fused receive preparation

The raw All-to-All receive buffer is not yet in the final attention layout.

A generic receive path would require separate unpacking, reshaping, permutation,
continuous materialization, and RoPE operators.

The CUDA receive-preparation path combines:

- rank-major receive-buffer decoding;
- Q/K/V plane extraction;
- head-sharded layout generation;
- RoPE application to Q and K;
- V layout preparation;
- optional F16 input and output handling.

```text
All-to-All receive buffer
        ↓ fused receive preparation
Q with RoPE
K with RoPE
V in attention layout
        ↓
SDPA
```

Bundled variants can prepare multiple Q/K/V planes from the same receive buffer
without independently repeating its indexing and layout conversion.

For FLUX double-stream attention, paired text and image streams can be decoded
and assembled directly into the joint attention sequence.

#### Packed reverse redistribution

After attention, the head-to-sequence path performs the inverse transformation.

The packed path combines:

- attention-output partitioning by destination sequence shard;
- head-to-sequence layout conversion;
- peer-major send-buffer generation;
- optional F16 communication;
- receive-side unpacking;
- direct F16 or BF16 local output generation.

```text
head-sharded attention output
        ↓ packed head-to-sequence operator
peer-major communication buffer
        ↓ All-to-All
sequence-sharded local output
```

Directly producing the dtype and layout required by the following DiT operator
reduces post-communication casts and contiguous copies.

Unsupported shapes or dtypes fall back to graph compositions built from
generic ggml layout operators.

Sequence parallelism is workload-sensitive. It is most effective when the
sequence length, attention cost, or activation-memory requirement is large
enough to offset:

- two All-to-All collectives per attention block;
- communication-buffer preparation;
- synchronization;
- graph segmentation and launch overhead.

For smaller image workloads or low-step generation, these overheads may exceed
the reduction in local computation. Sequence parallelism is therefore both a
memory-scaling mechanism and a workload-dependent performance optimization.

## Related Documentation

- [performance (RTX 4090)](../performance-4090.md)
- [performance (H200): Parallel results](../performance-H200.md)
- [Graph and operator optimization](graph-and-operator-optimization.md)
- [Build and installation](../build.md)
- [Command line usage](../cli.md)
