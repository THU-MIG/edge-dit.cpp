"""ctypes 封装 edge 编好的 libggml,直接调用真实的 q4_K 量化/反量化路径。
用于离线验证 imatrix 对 q4_K 量化误差的影响。不改任何 C++。
"""
import ctypes
import os
import numpy as np

# ggml_type 枚举 (third_party/ggml/include/ggml.h)
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q4_K = 12
QK_K = 256

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_LIBDIR = os.environ.get("EDGE_DIT_LIBDIR", os.path.join(_REPO_ROOT, "build-cuda", "bin"))


class GGMLQuant:
    def __init__(self, libdir=DEFAULT_LIBDIR):
        # 按依赖加载。base 含 ggml_quantize_chunk / dequantize_row_q4_K
        self.base = ctypes.CDLL(
            os.path.join(libdir, "libggml-base.so"), mode=ctypes.RTLD_GLOBAL
        )

        self.base.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.base.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,                    # type
            ctypes.POINTER(ctypes.c_float),  # src (f32)
            ctypes.c_void_p,                 # dst
            ctypes.c_int64,                  # start
            ctypes.c_int64,                  # nrows
            ctypes.c_int64,                  # n_per_row
            ctypes.POINTER(ctypes.c_float),  # imatrix (n_per_row 长, 可 NULL)
        ]

        self.base.ggml_row_size.restype = ctypes.c_size_t
        self.base.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]

        # dequantize_row_q4_K(const block_q4_K* x, float* y, int64_t k)
        self.base.dequantize_row_q4_K.restype = None
        self.base.dequantize_row_q4_K.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int64,
        ]

    def quantize_q4_K(self, W, imatrix=None):
        """W: [nrows, n_per_row] float32。imatrix: 长 n_per_row 的 per-input-channel 重要性,
        None 表示走 ref 路径(等价 edge 的全1.0? 见说明)。返回 dst 缓冲(bytes)、nrows、n_per_row。

        注意: edge 传的是全 1.0 向量(非 NULL),走 quantize_row_q4_K_impl;
        NULL 走 quantize_row_q4_K_ref(内部用 sigma 加权,与全1.0 impl 不同)。
        为忠实复现 edge 的"全1.0 基线",本函数默认要求显式传 imatrix。
        """
        assert W.ndim == 2
        nrows, n_per_row = W.shape
        assert n_per_row % QK_K == 0, f"n_per_row={n_per_row} 非 {QK_K} 倍数"
        Wc = np.ascontiguousarray(W, dtype=np.float32)
        row_size = self.base.ggml_row_size(GGML_TYPE_Q4_K, n_per_row)
        dst = ctypes.create_string_buffer(int(row_size) * int(nrows))

        if imatrix is None:
            im_ptr = ctypes.POINTER(ctypes.c_float)()  # NULL
        else:
            im = np.ascontiguousarray(imatrix, dtype=np.float32)
            assert im.size == n_per_row, f"imatrix 长 {im.size} != n_per_row {n_per_row}"
            im_ptr = im.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

        ret = self.base.ggml_quantize_chunk(
            GGML_TYPE_Q4_K,
            Wc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dst,
            0,
            int(nrows),
            int(n_per_row),
            im_ptr,
        )
        assert ret == int(row_size) * int(nrows), f"quantize 返回 {ret} 异常"
        return dst, nrows, n_per_row

    def dequantize_q4_K(self, dst, nrows, n_per_row):
        n = int(nrows) * int(n_per_row)
        out = np.empty(n, dtype=np.float32)
        self.base.dequantize_row_q4_K(
            ctypes.cast(dst, ctypes.c_void_p),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        return out.reshape(nrows, n_per_row)

    def roundtrip_q4_K(self, W, imatrix=None):
        """量化再反量化,返回 q4_K 重建的 f32 权重(与 W 同 shape)。"""
        dst, nrows, n_per_row = self.quantize_q4_K(W, imatrix)
        return self.dequantize_q4_K(dst, nrows, n_per_row)
