# SP Benchmark Report

本报告只记录 Sequence Parallel 的性能收益。速度采用模型日志里的 sampling hot path 时间，不把模型加载、文本编码、VAE decode、文件保存等端到端开销算进 speedup。

## Test Setup

```text
Date: 2026-06-13
GPU: NVIDIA H200
edge-dit HEAD: 939b8abf27dc70cae3b43d950653c50f6e9f30d8
ggml HEAD: e15ce1439100e856241a89661c5e7a8af3123281
Devices:
  1 GPU  -> 3
  2 GPUs -> 3,5
  4 GPUs -> 3,4,5,6
Image steps: 2
Wan steps: 2
Wan frames: 40
Seed: 0
```

## SD3 Medium

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/stable-diffusion-3-medium-diffusers \
  -p "a cute cat holding a white sign with the exact text 'sd3.cpp' written clearly on it" \
  -W <W> -H <H> --steps 2 -s 0 \
  --cfg-scale 5.0 --flow-shift 3.0 \
  --devices <devices> [--sp-size <n>] \
  -o <output>.png
```

| resolution | 1 GPU | 2 GPUs | 2 GPU speedup | 4 GPUs | 4 GPU speedup |
|---:|---:|---:|---:|---:|---:|
| 512x512 | 0.16s | 0.19s | 0.84x | 0.18s | 0.89x |
| 1024x1024 | 0.54s | 0.40s | 1.35x | 0.33s | 1.64x |
| 2048x2048 | 4.52s | 2.75s | 1.64x | 1.71s | 2.64x |

结论：SD3 在 512 档没有收益，1024 开始收益明显，2048 四卡达到 2.64x。

## Flux Dev

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/flux-dev \
  -p "a cinematic photo of a glass teapot on a wooden table, soft morning light" \
  -W <W> -H <H> --steps 2 -s 0 \
  --guidance 3.5 \
  --devices <devices> [--sp-size <n>] \
  -o <output>.png
```

| resolution | 1 GPU | 2 GPUs | 2 GPU speedup | 4 GPUs | 4 GPU speedup |
|---:|---:|---:|---:|---:|---:|
| 512x512 | 0.31s | 0.53s | 0.58x | 0.48s | 0.65x |
| 1024x1024 | 1.21s | 1.01s | 1.20x | 0.73s | 1.66x |
| 2048x2048 | 8.39s | 4.98s | 1.68x | 2.70s | 3.11x |

结论：Flux 的小分辨率 SP 开销较重，2048 档收益显著，四卡达到 3.11x。

## Qwen-Image

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/Qwen-Image \
  -p "a cute cat holding a white sign with the exact text 'qwen image' written clearly on it" \
  -W <W> -H <H> --steps 2 -s 0 \
  --devices <devices> [--sp-size <n>] \
  -o <output>.png
```

| resolution | 1 GPU | 2 GPUs | 2 GPU speedup | 4 GPUs | 4 GPU speedup |
|---:|---:|---:|---:|---:|---:|
| 512x512 | 0.35s | 0.80s | 0.44x | 0.74s | 0.47x |
| 1024x1024 | 1.63s | 1.50s | 1.09x | 1.11s | 1.47x |
| 2048x2048 | 14.62s | 7.78s | 1.88x | 4.43s | 3.30x |

结论：Qwen-Image 对分辨率更敏感，2048 档四卡达到 3.30x。

## Wan2.1 T2V 1.3B

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --video \
  --model /path/to/Wan2.1-T2V-1.3B-Diffusers \
  -p "a small robot walking through a rainy neon street, cinematic lighting" \
  -W <W> -H <H> --frames 40 --fps 16 --steps 2 -s 0 \
  --cfg-scale 5.0 --flow-shift 5.0 \
  --devices <devices> [--sp-size <n>] \
  -o <output>.mp4
```

| resolution | frames | 1 GPU | 2 GPUs | 2 GPU speedup | 4 GPUs | 4 GPU speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 416x240 | 40 | 0.71s | 0.52s | 1.37x | 0.42s | 1.69x |
| 640x384 | 40 | 2.47s | 1.20s | 2.06x | 0.78s | 3.17x |
| 832x480 | 40 | 5.28s | 2.31s | 2.29x | 1.23s | 4.29x |

结论：Wan 的序列更长，SP 收益最明显。两卡超过 2x 的结果如实记录；长序列 attention 计算随 sequence 近似二次增长，切分 sequence 后 hot path 超过线性直觉是可以理解的。

## Summary

| model | best tested resolution | best 2 GPU speedup | best 4 GPU speedup |
|---|---:|---:|---:|
| SD3 Medium | 2048x2048 | 1.64x | 2.64x |
| Flux Dev | 2048x2048 | 1.68x | 3.11x |
| Qwen-Image | 2048x2048 | 1.88x | 3.30x |
| Wan2.1 T2V 1.3B | 832x480x40f | 2.29x | 4.29x |

## Reproduce

```bash
python3 scripts/benchmark_sp_matrix.py \
  --models sd3,flux,qwen,wan \
  --image-resolutions 512x512,1024x1024,2048x2048 \
  --video-resolutions 416x240,640x384,832x480 \
  --gpu-groups '1:3;2:3,5;4:3,4,5,6' \
  --image-steps 2 \
  --video-steps 2 \
  --video-frames 40 \
  --out-dir /tmp/edge_dit_sp_benchmark \
  --report docs/sp_benchmark_report.md
```

原始结果：

```text
Image results:      /tmp/edge_dit_sp_image_4scale_20260613/results.json
Wan 416/832:        /tmp/edge_dit_sp_benchmark_20260613/results.json
Wan 640x384:        /tmp/edge_dit_sp_benchmark_wan_640x384_20260613/results.json
```
