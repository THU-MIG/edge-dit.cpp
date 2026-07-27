"""imatrix 离线校准工具(纯 Python,不改 C++)。

收集每层每输入通道的激活均方 E[x^2] 作为量化重要性(imatrix),喂给 q4_K 等
k-quant 的加权取整。重要性度量借用了 AWQ 的思想(激活大=权重列重要),但这里
只产出 imatrix 加权,并未实现 AWQ 的 per-channel scaling。

用 diffusers 加载 SD3-medium,对 transformer(DiT)的每个 nn.Linear 注册 forward hook,
跑若干多样 prompt 的去噪前向,收集每层输入激活的 per-input-channel 统计:
    - sum_x2[c] = sum over all tokens of x[:,c]^2   (→ E[x^2] 均方激活)
这个 E[x^2] 度量借用了 AWQ 的显著性思想(激活大 = 权重列重要),也是 GPTQ Hessian(X X^T)的对角近似。
由此得到的 per-input-channel 重要性向量,长度 = in_features = edge 里的 n_per_row,
正好喂进 edge 的 ggml_quantize_chunk(imatrix 接口),运行时零改动。

同时对每层保留一小批真实激活样本(reservoir),用于离线验证时计算"输出域误差"
(||W X - Wq X|| / ||W X||)——这是画质最直接的代理。

产物写到 --outdir(默认当前目录下的 imatrix-out/):
    imatrix.npz   每层 imatrix 向量(float32) + count
    imatrix.gguf  规范 gguf 格式(面向未来 C++ 集成)
    acts.npz      每层激活样本(float16, 每层 <=SAMPLE_ROWS 行)
"""
import os
import sys
import argparse

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import torch

def log(*a):
    print(*a, flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to the model (diffusers dir or safetensors), e.g. /path/to/sd3-medium")
    ap.add_argument("--outdir", default="imatrix-out",
                    help="calibration outputs (imatrix.gguf etc), default ./imatrix-out")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--nprompts", type=int, default=16)
    ap.add_argument("--sample-rows", type=int, default=256, help="每层保留的激活样本行数")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.nprompts, args.sample_rows = 2, 2, 64
    os.makedirs(args.outdir, exist_ok=True)

    from diffusers import StableDiffusion3Pipeline

    dev = "cuda"
    log(f"[load] {args.model} steps={args.steps} nprompts={args.nprompts}")
    pipe = StableDiffusion3Pipeline.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to(dev)
    tf = pipe.transformer
    tf.eval()

    # ---- 注册 hook: 收集每个 nn.Linear 的输入激活统计 ----
    stats = {}   # name -> dict(sum_x2 float64[in], count int, sample float16[R,in])
    handles = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0]
            if x is None:
                return
            xf = x.detach().reshape(-1, x.shape[-1]).float()  # [N, in]
            s = stats.get(name)
            sx2 = (xf * xf).sum(dim=0).double().cpu().numpy()  # [in]
            n = xf.shape[0]
            if s is None:
                # 初始化 + 首批做激活样本(取前 SAMPLE_ROWS 行)
                r = min(args.sample_rows, xf.shape[0])
                stats[name] = {
                    "sum_x2": sx2,
                    "count": int(n),
                    "in": int(xf.shape[1]),
                    "sample": xf[:r].half().cpu().numpy(),
                }
            else:
                s["sum_x2"] += sx2
                s["count"] += int(n)
        return hook

    n_lin = 0
    for name, mod in tf.named_modules():
        if isinstance(mod, torch.nn.Linear):
            handles.append(mod.register_forward_hook(make_hook(name)))
            n_lin += 1
    log(f"[hook] 注册 {n_lin} 个 nn.Linear")

    # ---- 多样校准 prompt ----
    base_prompts = [
        "a photograph of an astronaut riding a horse on the moon",
        "a bustling medieval marketplace at golden hour, highly detailed",
        "close-up portrait of an elderly fisherman, weathered skin, studio light",
        "a serene japanese zen garden with cherry blossoms, misty morning",
        "a futuristic cyberpunk city street at night, neon reflections, rain",
        "a bowl of ramen with steam, food photography, shallow depth of field",
        "an oil painting of a stormy sea with a lighthouse",
        "a cute corgi puppy playing in autumn leaves, bokeh",
        "architectural render of a modern glass museum, blue sky",
        "a fantasy dragon perched on a snowy mountain peak, epic lighting",
        "macro shot of a dewdrop on a spider web at dawn",
        "a vintage red sports car parked on a coastal road",
        "abstract geometric pattern, vibrant colors, bauhaus style",
        "a wizard casting a glowing spell in a dark forest, cinematic",
        "flat lay of watercolor art supplies on a wooden desk",
        "a majestic tiger walking through tall grass, wildlife photography",
    ]
    prompts = base_prompts[:args.nprompts]

    gen = torch.Generator(device=dev).manual_seed(1234)
    with torch.no_grad():
        for i, p in enumerate(prompts):
            _ = pipe(
                p,
                num_inference_steps=args.steps,
                guidance_scale=7.0,
                height=1024, width=1024,
                generator=gen,
                output_type="latent",
            )
            log(f"[calib] {i+1}/{len(prompts)} done: {p[:40]}")

    for h in handles:
        h.remove()

    # ---- 产出 imatrix: E[x^2] = sum_x2 / count ----
    imatrix = {}
    acts = {}
    meta = {}
    for name, s in stats.items():
        ex2 = (s["sum_x2"] / max(1, s["count"])).astype(np.float32)  # [in]
        imatrix[name] = ex2
        acts[name] = s["sample"]
        meta[name] = np.array([s["count"], s["in"]], dtype=np.int64)

    np.savez(os.path.join(args.outdir, "imatrix.npz"), **imatrix,
             **{f"__meta__{k}": v for k, v in meta.items()})
    np.savez(os.path.join(args.outdir, "acts.npz"), **acts)
    log(f"[save] imatrix.npz / acts.npz : {len(imatrix)} 层")

    # ---- 规范 gguf(面向未来 C++ 集成),每层一个 float32 tensor ----
    try:
        import gguf
        w = gguf.GGUFWriter(os.path.join(args.outdir, "imatrix.gguf"), "sd3-dit-imatrix")
        w.add_uint32("imatrix.n_tensors", len(imatrix))
        w.add_string("imatrix.source", "sd3-medium Ex2 activation calibration")
        for name, v in imatrix.items():
            # gguf tensor 名: 用 diffusers 权重名 + .weight(C++ 集成时映射到 edge tensor 名)
            w.add_tensor(name + ".weight", v.reshape(1, -1))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        log("[save] imatrix.gguf")
    except Exception as e:
        log(f"[warn] gguf 导出失败(不影响验证): {e}")

    # 打印几个代表层的 imatrix 分布,直观看离群通道
    for name in list(imatrix.keys())[:3] + [k for k in imatrix if "attn.to_q" in k][:1]:
        v = imatrix[name]
        log(f"[imatrix] {name}: in={v.size} max/mean={v.max()/v.mean():.1f} "
            f"top1%/median={np.percentile(v,99)/ (np.median(v)+1e-12):.1f}")
    log("[done] calibrate")

if __name__ == "__main__":
    main()
