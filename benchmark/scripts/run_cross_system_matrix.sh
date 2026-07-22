#!/usr/bin/env bash
# 跨系统对比矩阵总控脚本: 8卡并行出图 -> 质量评测(按系统分片) -> 聚合出表(全指标)
#
# 前置: 先激活好带评测依赖的 Python 环境(见 benchmark/requirements.txt), 例如
#   conda activate edge
# 然后:
#   nohup bash benchmark/scripts/run_cross_system_matrix.sh > benchmark/results/matrix.log 2>&1 &
#
# 可选环境变量(不设则用合理默认, 无机器专属硬编码):
#   ED_BENCH_PYTHON    评测用的 python(默认 PATH 里的 python, 即已激活环境)
#   ED_BENCH_CUDNN_LIB cudnn 库目录(默认从 pip 安装的 nvidia-cudnn 包动态推导)
#   ED_BENCH_SITE      site 配置(默认 benchmark/configs/local/site-4090.yaml)
#   HF_ENDPOINT        HF 镜像(默认 https://hf-mirror.com)
#
# 一次跑通: 出图(T2I/edit/video) -> eval_all.py(按任务正确路由所有指标, 模型只加载一次,
# 质量分回写 result.json + 每run eval/quality.json) -> make_matrix_tables.py(每配置3prompt
# 明细+均值, 组件耗时/显存/端到端(标口径)/CLIP/美学/IR/量化vsFP16 + edit/video专属指标).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

PY="${ED_BENCH_PYTHON:-python}"
# cudnn 库路径: 优先用环境变量, 否则从 pip 安装的 nvidia-cudnn 包动态推导(无则留空)
CUDNN_LIB="${ED_BENCH_CUDNN_LIB:-$("$PY" -c 'import os,nvidia.cudnn as c; print(os.path.join(os.path.dirname(c.__file__),"lib"))' 2>/dev/null || true)}"
export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu${CUDNN_LIB:+:$CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

SITE="${ED_BENCH_SITE:-benchmark/configs/local/site-4090.yaml}"
OUTPUT_ROOT="benchmark/results/cross-system-matrix"

# 8个suite分配到8张卡(卡号互斥,单run独占卡满足组件级显存归因)。
SUITES=(
  "xsys-edge"            # GPU0
  "xsys-sdcpp"           # GPU1
  "xsys-diffusers"       # GPU2
  "xsys-edit-edge"       # GPU3
  "xsys-edit-sdcpp"      # GPU4
  "xsys-edit-diffusers"  # GPU5
  "xsys-video-edge"      # GPU6
  "xsys-video-diffusers" # GPU7
)

echo "=================================================="
echo "阶段1: 8卡并行出图 (每suite绑一张卡, --resume)"
echo "开始: $(date)"
echo "=================================================="
mkdir -p "$OUTPUT_ROOT"
pids=()
for gpu in "${!SUITES[@]}"; do
  suite="${SUITES[$gpu]}"
  logf="$OUTPUT_ROOT/run_${suite}.log"
  echo "GPU$gpu <- $suite (log: $logf)"
  BENCHMARK_CUDA_VISIBLE_DEVICES=$gpu \
    "$PY" benchmark/orchestration/run_suite.py \
      --suite "benchmark/configs/suites/${suite}.yaml" \
      --site "$SITE" \
      --execute --resume \
      --output-root "$OUTPUT_ROOT" > "$logf" 2>&1 &
  pids+=($!)
done
echo "等待8个出图进程..."
for pid in "${pids[@]}"; do wait "$pid"; done
echo "阶段1完成: $(date)"

echo "=================================================="
echo "阶段2: 质量评测 (eval_all.py, 按系统分3卡并行)"
echo "  路由: T2I=CLIP/美学/IR; edit=方向CLIP/保持度/美学/IR; video=逐帧+时序一致性"
echo "  量化vsFP16(PSNR/SSIM/LPIPS)在同系统内配对, 故按系统分片不破坏配对口径"
echo "=================================================="
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
epids=()
i=0
for system in "edge-dit.cpp" "diffusers" "stable-diffusion.cpp"; do
  logf="$OUTPUT_ROOT/eval_${i}.log"
  echo "GPU$i <- eval $system (log: $logf)"
  CUDA_VISIBLE_DEVICES=$i \
    "$PY" benchmark/scripts/eval_all.py \
      --results-root "$OUTPUT_ROOT" \
      --site "$SITE" \
      --only-system "$system" \
      --device cuda:0 > "$logf" 2>&1 &
  epids+=($!)
  i=$((i+1))
done
echo "等待3个评测进程..."
for pid in "${epids[@]}"; do wait "$pid"; done
echo "阶段2完成: $(date)"

echo "=================================================="
echo "阶段3: 聚合出表 (全指标一次性 markdown)"
echo "=================================================="
"$PY" benchmark/scripts/make_matrix_tables.py \
  --results-root "$OUTPUT_ROOT" \
  --output "$OUTPUT_ROOT/tables_matrix.md" 2>&1 || echo "出表失败"

echo "=================================================="
echo "全部完成: $(date)"
echo "结果: $OUTPUT_ROOT/tables_matrix.md (每run: result.json + eval/quality.json)"
echo "各run图/视频在各自 samples/ 目录 (画质请自行查看)"
echo "=================================================="
