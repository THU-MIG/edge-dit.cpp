"""全层(340 个 Linear)聚合验证: ones vs 激活校准 imatrix 的 q4_K 输出域误差。
按层类型分桶统计, 给出全局结论。"""
import os
import numpy as np
from ggml_quant import GGMLQuant
from safetensors import safe_open

DIR = os.environ.get("IMATRIX_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
SD3 = "/home/public/models/sd3-medium/transformer/diffusion_pytorch_model.fp16.safetensors"


def bucket(name):
    for key in ["attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
                "ff.net.0.proj", "ff.net.2", "ff_context.net.0", "ff_context.net.2",
                "norm1.linear", "norm1_context.linear", "norm.linear",
                "context_embedder", "time_text_embed", "proj_out"]:
        if key in name:
            return key
    return "other"


def main():
    q = GGMLQuant()
    im = np.load(os.path.join(DIR, "imatrix.npz"))
    acts = np.load(os.path.join(DIR, "acts.npz"))
    names = [k for k in im.files if not k.startswith("__meta__")]
    sf = safe_open(SD3, framework="np")
    sf_keys = set(sf.keys())

    per_bucket = {}   # bucket -> list of (oe_ones, oe_awq)
    tot = {"ones": [], "imx": []}
    n_ok = 0
    for name in names:
        wkey = name + ".weight"
        if wkey not in sf_keys or name not in acts.files:
            continue
        W = sf.get_tensor(wkey).astype(np.float32)
        if W.ndim != 2 or W.shape[1] % 256 != 0:
            continue
        ex2 = im[name].astype(np.float32)
        if ex2.size != W.shape[1]:
            continue
        X = acts[name].astype(np.float32)
        imx = ex2 / (ex2.mean() + 1e-12)
        ones = np.ones_like(ex2)

        Wq_o = q.roundtrip_q4_K(W, ones)
        Wq_a = q.roundtrip_q4_K(W, imx)
        Y = X @ W.T
        den = np.linalg.norm(Y) + 1e-12
        oe_o = float(np.linalg.norm(Y - X @ Wq_o.T) / den)
        oe_a = float(np.linalg.norm(Y - X @ Wq_a.T) / den)

        b = bucket(name)
        per_bucket.setdefault(b, []).append((oe_o, oe_a))
        tot["ones"].append(oe_o)
        tot["imx"].append(oe_a)
        n_ok += 1

    print(f"验证层数: {n_ok}")
    print(f"{'bucket':<24} {'n':>4} {'oErr_ones':>10} {'oErr_awq':>10} {'降幅%':>8}")
    print("-" * 60)
    for b in sorted(per_bucket, key=lambda k: -np.mean([x[0] for x in per_bucket[k]])):
        arr = np.array(per_bucket[b])
        mo, ma = arr[:, 0].mean(), arr[:, 1].mean()
        print(f"{b:<24} {len(arr):>4} {mo:>10.5f} {ma:>10.5f} {100*(1-ma/mo):>7.1f}%")
    print("-" * 60)
    mo, ma = np.mean(tot["ones"]), np.mean(tot["imx"])
    print(f"{'全体 mean':<24} {n_ok:>4} {mo:>10.5f} {ma:>10.5f} {100*(1-ma/mo):>7.1f}%")
    # 中位数(避免极端层主导)
    mo2 = np.median(tot["ones"]); ma2 = np.median(tot["imx"])
    print(f"{'全体 median':<24} {n_ok:>4} {mo2:>10.5f} {ma2:>10.5f} {100*(1-ma2/mo2):>7.1f}%")
    # 逐层配对: imatrix 更好的比例
    arr = np.array([tot["ones"], tot["imx"]])
    better = np.mean(arr[1] < arr[0]) * 100
    worse = np.mean(arr[1] > arr[0] * 1.01) * 100  # 明显变差(>1%)
    print(f"imatrix 逐层更优比例: {better:.1f}% ; 明显变差(>1%)比例: {worse:.1f}%")


if __name__ == "__main__":
    main()
