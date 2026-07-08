#ifndef __DIT_MODULATION_HPP__
#define __DIT_MODULATION_HPP__

#include "ggml.h"

namespace dit {

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* shift,
                      ggml_tensor* scale,
                      bool skip_reshape = false);

ggml_tensor* modulate(ggml_context* ctx,
                      ggml_tensor* x,
                      ggml_tensor* scale,
                      bool skip_reshape = false);

ggml_tensor* residual_gate(ggml_context* ctx,
                           ggml_tensor* residual,
                           ggml_tensor* x,
                           ggml_tensor* gate,
                           bool skip_reshape = false);

}  // namespace dit

#endif  // __DIT_MODULATION_HPP__
