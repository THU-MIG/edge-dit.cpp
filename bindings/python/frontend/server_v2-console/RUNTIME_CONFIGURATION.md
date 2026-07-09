# Runtime 配置指南

这份文档只讲一件事：

- 把 `server_v2` Web Console 迁到另一台设备时，
- 怎样用最少改动完成配置，
- 并且区分简单版和复杂版两种做法。

推荐顺序很简单：

1. 先用现成 profile
2. 先改环境变量
3. 跑不稳再加本地 profile

## 1. 当前哪些地方可以配置

当前实现里，运行配置主要来自两处：

- `runtime/profiles/*.json`
- 本地环境变量

它们的分工是：

- profile 负责模型类型和默认引擎参数
- 环境变量负责本机路径覆盖

最常用的脚本是：

- [scripts/runtime-env.sh](scripts/runtime-env.sh)
- [scripts/run-managed-profile.sh](scripts/run-managed-profile.sh)

## 2. 简单版：只改环境变量

这是最推荐的第一步。

适用情况：

- 模型没变
- 只是路径变了
- 新设备和当前验证机器差异不大

### 2.1 需要准备

先确认本机已经具备：

- 仓库代码
- `libedgedit.so`
- 可用的 Python
- CUDA / cuDNN 依赖
- 模型目录

### 2.2 需要设置的变量

先设置基础变量：

```bash
export EDGE_DIT_REPO_ROOT=/path/to/edge-dit.cpp
export EDGE_DIT_PYTHON_BIN=/usr/bin/python3
export EDGE_DIT_LIBRARY=/path/to/edge-dit.cpp/build-cuda-shared/bin/libedgedit.so
export EDGE_DIT_DEPENDENCY_DIRS=/path/to/cudnn/lib:/path/to/cuda_nvrtc/lib:/path/to/cublas/lib:/path/to/cuda_runtime/lib:/path/to/edge-dit.cpp/build-cuda-shared/bin
```

再设置模型路径变量。常见对应关系如下：

- `flux-dev` -> `EDGE_DIT_FLUX_MODEL_PATH`
- `flux-kontext` -> `EDGE_DIT_FLUX_KONTEXT_MODEL_PATH`
- `qwen-image` -> `EDGE_DIT_QWEN_IMAGE_MODEL_PATH`
- `qwen-image-edit` -> `EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH`
- `sd3-medium` -> `EDGE_DIT_SD3_MEDIUM_MODEL_PATH`
- `wan-t2v` -> `EDGE_DIT_WAN_VIDEO_MODEL_PATH`

例如：

```bash
export EDGE_DIT_FLUX_KONTEXT_MODEL_PATH=/models/FLUX.1-Kontext-dev
export EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH=/models/Qwen-Image-Edit
export EDGE_DIT_WAN_VIDEO_MODEL_PATH=/models/Wan2.1-T2V-1.3B-Diffusers
```

### 2.3 启动方式

在 `bindings/python/frontend/server_v2-console` 目录下运行：

```bash
npm run dev:managed
```

如果想直接指定启动某个 profile：

```bash
npm run dev:managed -- --auto-start-profile flux-kontext
```

### 2.4 前端里怎么操作

前端里不需要手写这些参数。

你只需要：

1. 打开控制台
2. 在 `Local Runtime` 里选择模型
3. 点击启动或切换

如果简单版能跑通，就不需要继续折腾。

## 3. 复杂版：增加本地 profile

只有下面这些情况，才建议进入复杂版：

- 新设备显存明显更小或更大
- 现成 profile 会 OOM
- 需要改卸载策略
- 需要改 `weight_type` 或 `max_vram_gb`

### 3.1 做法

在下面目录新增一个本地 profile 文件：

```text
bindings/python/frontend/server_v2-console/runtime/profiles/
```

推荐命名：

- `flux-kontext-12gb-local.json`
- `qwen-image-edit-24gb-local.json`

### 3.2 一个最小例子

```json
{
  "slug": "qwen-image-edit-12gb-local",
  "name": "Qwen-Image-Edit (12GB Local)",
  "kind": "image",
  "model_env": "EDGE_DIT_QWEN_IMAGE_EDIT_MODEL_PATH",
  "engine": {
    "model_path": "/models/Qwen-Image-Edit",
    "backend": "cuda",
    "weight_type": "q4_k",
    "offload_params_to_cpu": true,
    "keep_text_encoder_on_cpu": true,
    "keep_vae_on_cpu": true,
    "max_vram_gb": 12.0
  }
}
```

建议只改这些真正和机器资源相关的字段：

- `max_vram_gb`
- `offload_params_to_cpu`
- `keep_text_encoder_on_cpu`
- `keep_vae_on_cpu`
- `weight_type`
- `backend`

### 3.3 推荐规则

- 优先复制最接近的现成 profile
- `model_env` 尽量保持不变
- 不要把机器私有路径改进共享 profile
- 本地 profile 最好保持未跟踪，或加到 `.git/info/exclude`

## 4. 给新设备用户的完整流程

推荐按这个顺序做：

1. 先设置环境变量
2. 先复用现成 profile
3. 运行 `npm run dev:managed`
4. 在前端做一次最小 smoke test
5. 如果只是路径问题，到这里就结束
6. 如果遇到显存或加载策略问题，再创建本地 profile

一句话总结：

> 先用环境变量解决路径问题，再用本地 profile 解决机器差异问题。

## 5. 当前方案的边界

现在已经有：

- 现成 profile
- 环境变量覆盖模型路径
- runtime manager 按 profile 启动

现在还没有：

- 浏览器里直接编辑启动参数
- 自动保存本地 override
- 一键导入导出机器配置

所以当前最稳妥的做法仍然是：

> 简单版先跑通，复杂版再定制。
