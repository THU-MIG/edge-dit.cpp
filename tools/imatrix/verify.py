"""离线验证: 全1.0 imatrix (edge 现状基线) vs 激活校准 imatrix, 对 q4_K 量化误差的影响。

调用 edge 真实的 ggml_quantize_chunk (q4_K) + dequantize_row_q4_K (ctypes)。

两个层面的误差:
  (A) 权重域: MSE(W, dequant(quant(W, im)))  —— 整体 & 重要通道(高激活列)
  (B) 输出域: ||W X - Wq X||_F / ||W X||_F   —— 用校准收集的真实激活 X, 画质最直接代理

对比的 imatrix:
  ones : 全 1.0 (复现 edge model_loader.cpp 现状)
  imx  : E[x^2] (校准得到的 per-input-channel 激活均方, 借用 AWQ 显著性思想)
  imx_p: E[x^2]^alpha 的幂律平滑(温和缩放, 默认 alpha=0.5)

选取一批代表性 DiT Linear 层报告。
"""
import os
import argparse
import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from ggml_quant import GGMLQuant


def out_err(W, Wq, X):
    """输出域相对误差: X [N,in], W [out,in]. Y=X W^T."""
    Y = X @ W.T
    Yq = X @ Wq.T
    num = np.linalg.norm(Y - Yq)
    den = np.linalg.norm(Y) + 1e-12
    return float(num / den)


def w_mse(W, Wq, cols=None):
    if cols is None:
        return float(np.mean((W - Wq) ** 2))
    return float(np.mean((W[:, cols] - Wq[:, cols]) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(_SCRIPT_DIR, "out"))
    ap.add_argument("--sd3", default="/home/public/models/sd3-medium/transformer/diffusion_pytorch_model.fp16.safetensors")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--nlayers", type=int, default=14)
    args = ap.parse_args()

    q = GGMLQuant()
    im_npz = np.load(os.path.join(args.dir, "imatrix.npz"))
    acts = np.load(os.path.join(args.dir, "acts.npz"))
    names = [k for k in im_npz.files if not k.startswith("__meta__")]

    # 选代表层: 各类型都覆盖
    want_sub = ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "attn.add_q_proj", "ff.net.0.proj", "ff.net.2",
                "norm1.linear", "context_embedder", "time_text_embed"]
    picked = []
    for sub in want_sub:
        for n in names:
            if sub in n and n not in picked:
                picked.append(n)
                break
    # 补齐到 nlayers
    for n in names:
        if len(picked) >= args.nlayers:
            break
        if n not in picked:
            picked.append(n)
    picked = picked[:args.nlayers]

    from safetensors import safe_open
    sf = safe_open(args.sd3, framework="np")
    sf_keys = set(sf.keys())

    print(f"{'layer':<42} {'in':>5} {'oErr_ones':>10} {'oErr_awq':>10} {'oErr_awqP':>10} "
          f"{'wMSEsal_ones':>12} {'wMSEsal_awq':>12} {'gain%':>7}")
    print("-" * 118)

    agg = {"ones": [], "imx": [], "imxp": []}
    agg_sal = {"ones": [], "imx": []}
    for name in picked:
        wkey = name + ".weight"
        if wkey not in sf_keys:
            continue
        W = sf.get_tensor(wkey).astype(np.float32)  # [out, in]
        if W.ndim != 2 or W.shape[1] % 256 != 0:
            continue
        ex2 = im_npz[name].astype(np.float32)        # [in]
        if ex2.size != W.shape[1]:
            continue
        X = acts[name].astype(np.float32) if name in acts.files else None

        ones = np.ones_like(ex2)
        imx = ex2.copy()
        # 归一到均值1(不改变加权MSE的相对权重, 只为数值稳定)
        imx = imx / (imx.mean() + 1e-12)
        imxp = np.power(ex2 / (ex2.mean() + 1e-12), args.alpha)

        Wq_ones = q.roundtrip_q4_K(W, ones)
        Wq_imx = q.roundtrip_q4_K(W, imx)
        Wq_imxp = q.roundtrip_q4_K(W, imxp)

        # 重要通道 = 激活 top 5%
        thr = np.percentile(ex2, 95)
        sal = np.where(ex2 >= thr)[0]

        wmse_sal_ones = w_mse(W, Wq_ones, sal)
        wmse_sal_imx = w_mse(W, Wq_imx, sal)

        if X is not None:
            oe_ones = out_err(W, Wq_ones, X)
            oe_imx = out_err(W, Wq_imx, X)
            oe_imxp = out_err(W, Wq_imxp, X)
            agg["ones"].append(oe_ones)
            agg["imx"].append(oe_imx)
            agg["imxp"].append(oe_imxp)
        else:
            oe_ones = oe_imx = oe_imxp = float("nan")

        agg_sal["ones"].append(wmse_sal_ones)
        agg_sal["imx"].append(wmse_sal_imx)
        gain = 100.0 * (oe_ones - min(oe_imx, oe_imxp)) / (oe_ones + 1e-12)
        print(f"{name[:42]:<42} {W.shape[1]:>5} {oe_ones:>10.5f} {oe_imx:>10.5f} "
              f"{oe_imxp:>10.5f} {wmse_sal_ones:>12.3e} {wmse_sal_imx:>12.3e} {gain:>6.1f}%")

    def gm(x):
        x = np.array(x)
        return float(np.exp(np.mean(np.log(x + 1e-12))))
    print("-" * 118)
    if agg["ones"]:
        print(f"输出域相对误差 均值:  ones={np.mean(agg['ones']):.5f}  "
              f"imx={np.mean(agg['imx']):.5f}  imxP={np.mean(agg['imxp']):.5f}")
        r_imx = 100 * (1 - np.mean(agg["imx"]) / np.mean(agg["ones"]))
        r_imxp = 100 * (1 - np.mean(agg["imxp"]) / np.mean(agg["ones"]))
        print(f"  -> imx 相对 ones 降 {r_imx:+.1f}% , imxP 降 {r_imxp:+.1f}% (正=更好)")
    print(f"重要通道权重MSE 均值: ones={np.mean(agg_sal['ones']):.3e}  imx={np.mean(agg_sal['imx']):.3e}"
          f"  -> 降 {100*(1-np.mean(agg_sal['imx'])/np.mean(agg_sal['ones'])):+.1f}%")


if __name__ == "__main__":
    main()
