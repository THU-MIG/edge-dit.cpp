# edge-dit.cpp

`edge-dit.cpp` 是一个基于 C/C++ 的 DiT 推理项目，提供 `ed-cli` 命令行工具运行 SD3、Flux、Wan、Qwen-Image 等模型。

## 支持的模型

- **SD3**（stable-diffusion-3-medium-diffusers）
- **Flux1**（flux-dev、FLUX.1-Kontext-dev）
- **Qwen-Image**（qwen-image、qwen-image-edit）
- **Wan2.2-T2V-A14B**（Wan2.2-T2V-A14B-Diffusers，文生视频）
- **Wan2.1-T2V-1.3B**（Wan2.1-T2V-1.3B-Diffusers，文生视频）

## 编译

```bash
bash ./scripts/build_cuda.sh
bash ./scripts/build_cpu.sh
```

CUDA 构建脚本默认会启用 cuDNN SDPA fast attention。脚本会先查找
`CUDNN_ROOT` 或当前 Python/conda 环境中的 NVIDIA cuDNN wheel；如果没有找到，
会尝试用用户态 pip 安装 CUDA 12 cuDNN/NVRTC/runtime wheel，不需要 sudo：

```bash
bash ./scripts/build_cuda.sh
```

常用覆盖项：

```bash
# 使用已有 cuDNN 安装
CUDNN_ROOT=/path/to/nvidia/cudnn bash ./scripts/build_cuda.sh

# 禁止脚本自动 pip install cuDNN wheel，找不到 cuDNN 时退回普通 CUDA 构建
ED_INSTALL_CUDNN=OFF bash ./scripts/build_cuda.sh

# 完全关闭 cuDNN SDPA fast attention
ED_ENABLE_CUDNN_SDPA=OFF bash ./scripts/build_cuda.sh
```

编译完成后，命令行程序位于：

```bash
./build-cuda/bin/ed-cli   # GPU 推理
./build-cpu/bin/ed-cli    # CPU 推理
```

## 基本用法

### 使用 diffusers 目录加载模型

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/diffusers-model-dir \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 20 -s 0 \
  -o output.png
```

### 组件式路径加载（仅Flux和SD3支持）

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --diffusion-model /path/to/transformer.safetensors \
  --clip_l /path/to/clip_l.safetensors \
  --clip_g /path/to/clip_g.safetensors \
  --t5xxl /path/to/t5xxl.safetensors.index.json \
  --vae /path/to/vae.safetensors \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 20 -s 0 \
  -o output.png
```

### 视频生成（Wan）

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --video \
  --model /path/to/Wan2.2-T2V-A14B-Diffusers \
  -p "a small robot walking through a rainy neon street" \
  -W 832 -H 480 --frames 81 --fps 16 --steps 50 -s 0 \
  --cfg-scale 5.0 --flow-shift 5.0 \
  -o output.avi
```

## 量化与内存优化

### 在线量化（`--type`）

加载时实时量化权重，降低显存占用：

```bash
# Q8 量化（显存约为 FP16 的 56%）
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --type q8_0 \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png

# Q4_K 量化（显存约为 FP16 的 33%）
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --type q4_k \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png

# 混合量化：主体 Q4_K，norm 层保持 F16
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --type q4_k \
  --tensor-type-rules "norm=f16,bias=f32" \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png
```

支持的量化类型：`f32`、`f16`、`bf16`、`q4_0`、`q4_1`、`q5_0`、`q5_1`、`q8_0`、`q2_k`、`q3_k`、`q4_k`、`q5_k`、`q6_k`

### 跳过 T5（`--no-t5`，仅 SD3）

SD3 的 T5XXL 占 ~9 GB 显存但非必需：

```bash
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --no-t5 \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png
```

### VAE Tiling（`--vae-tiling`）

大分辨率图像的 VAE 解码分块执行，降低显存峰值：

```bash
# 启用 2×2 分块（省约 21% 显存）
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --vae-tiling \
  -p "a cat" -W 2048 -H 2048 --steps 20 -o output.png

# 4×4 更细分块（省约 28% 显存）
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --vae-tile-size 4 \
  -p "a cat" -W 2048 -H 2048 --steps 20 -o output.png
```

### 显存 Offload

将模型权重保留在 CPU，降低 GPU 常驻显存：

```bash
# 文本编码器放 CPU（省约 10 GB）
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --keep-text-encoder-on-cpu --vae-tiling \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png

# 全部 offload + 限制计算图显存
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 --offload-to-cpu --max-vram 6 --vae-tiling \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png
```

### 组合使用（小显存设备）

```bash
./build-cuda/bin/ed-cli --backend cuda --model /path/to/sd3 \
  --type q4_k --no-t5 --vae-tiling --keep-text-encoder-on-cpu \
  -p "a cat" -W 1024 -H 1024 --steps 20 -o output.png
```

## 并行推理

### CFG 并行（双 GPU）

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/model \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 20 -s 0 \
  --cfg-scale 5.0 \
  --devices 0,1 \
  --cfg-size 2 \
  -o output.png
```

### Sequence Parallel（多 GPU）

```bash
./build-cuda/bin/ed-cli --backend cuda \
  --model /path/to/model \
  -p "prompt text" \
  -W 1024 -H 1024 --steps 20 -s 0 \
  --devices 0,1 \
  --sp-size 2 \
  -o output.png
```

## 常用参数

### 模型加载
| 参数 | 说明 |
|---|---|
| `--backend cuda\|cpu` | 计算后端 |
| `--model <dir>` | diffusers 模型目录 |
| `--diffusion-model <path>` | 单独指定 transformer 权重 |
| `--clip_l <path>` | CLIP-L 权重 |
| `--clip_g <path>` | CLIP-G 权重 |
| `--t5xxl <path>` | T5XXL 权重 |
| `--vae <path>` | VAE 权重 |

### 生成参数
| 参数 | 说明 |
|---|---|
| `-p, --prompt <text>` | 文本提示词 |
| `-W, --width <int>` | 输出宽度，默认 1024 |
| `-H, --height <int>` | 输出高度，默认 1024 |
| `--steps <int>` | 采样步数，默认 20 |
| `-s, --seed <int64>` | 随机种子 |
| `--cfg-scale <float>` | CFG scale |
| `--guidance <float>` | Flux distilled guidance |
| `--flow-shift <float>` | Flow scheduler shift |
| `-i, --image <path>` | 图像编辑输入图；FLUX.1-Kontext-dev 将其作为参考图 |
| `--video` | 生成视频 |
| `--frames <int>` | 视频帧数 |
| `--fps <int>` | 视频帧率 |
| `-o, --output <path>` | 输出文件路径 |

### 量化与内存
| 参数 | 说明 |
|---|---|
| `--type <dtype>` | 在线量化类型 |
| `--tensor-type-rules <csv>` | 混合量化规则 |
| `--no-t5` | 跳过 T5 文本编码器（仅 SD3） |
| `--vae-tiling` | 启用 VAE 分块解码 |
| `--vae-tile-size <float>` | VAE 分块相对大小 |
| `--offload-to-cpu` | 全部权重放 CPU |
| `--keep-text-encoder-on-cpu` | 文本编码器放 CPU |
| `--keep-vae-on-cpu` | VAE 放 CPU |
| `--max-vram <GB>` | 计算图显存上限（配合 offload） |

### 并行
| 参数 | 说明 |
|---|---|
| `--devices <csv>` | GPU 列表，如 `0,1` |
| `--cfg-size <n>` | CFG 并行规模（1 或 2） |
| `--sp-size <n>` | Sequence parallel 规模 |
| `-t, --threads <int>` | CPU 线程数 |

## 更多文档

- 改进详情：[docs/improvements.md](docs/improvements.md)
- SP 性能报告：[docs/sp_benchmark_report.md](docs/sp_benchmark_report.md)
