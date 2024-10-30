import torch

import triton
import triton.language as tl


@triton.jit
def kernel(X, i: tl.constexpr):
    tl.store(X, i)

x = torch.empty(1, dtype=torch.int32, device='xpu')
h = kernel[(1, )](x, i=12)
assert x[0] == 12
# dis = h.asm["spvdis"]
