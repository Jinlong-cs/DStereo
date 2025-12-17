import torch
from torch.autograd import Function


class Interp1d(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, xp, fp):
        left = fp[0]
        right = fp[-1]

        N, C, H, W = x.shape
        x_flatten = x.view(N, -1)

        i = torch.clip(
            torch.searchsorted(xp, x_flatten, right=True), 1, len(xp) - 1
        )

        answer = torch.where(
            x_flatten < xp[0],
            left,
            (fp[i - 1] * (xp[i] - x_flatten) + fp[i] * (x_flatten - xp[i - 1]))
            / (xp[i] - xp[i - 1]),
        )
        answer = torch.where(x_flatten > xp[-1], right, answer)
        answer = answer.view(N, C, H, W)
        ctx.save_for_backward(i, x, xp, fp)
        return answer

    @staticmethod
    def backward(ctx, grad_out):
        i, x, xp, fp = ctx.saved_tensors
        N = x.shape[0]
        x_flatten = x.view(N, -1).clone()
        dxp = xp[i] - xp[i - 1]

        grad_x = grad_out.view(N, -1).clone()

        grad_x[x_flatten < xp[0]] = 0
        grad_x[x_flatten > xp[-1]] = 0

        dfp_i_minus_1 = grad_x * (xp[i] - x_flatten) / dxp
        dfp_i = grad_x * (x_flatten - xp[i - 1]) / dxp

        grad_fp = torch.zeros_like(fp)
        # Update the gradient for i-1 indices
        i_minus_1_indices = i - 1
        i_minus_1_indices = i_minus_1_indices.view(-1)
        grad_fp.scatter_add_(0, i_minus_1_indices, dfp_i_minus_1.view(-1))

        # Update the gradient for i indices
        i_indices = i
        i_indices = i_indices.view(
            -1,
        )
        grad_fp.scatter_add_(0, i_indices, dfp_i.view(-1))

        return None, None, grad_fp


class Quantize(Function):
    @staticmethod
    def forward(ctx, x, output_bit):
        x = x * (2 ** output_bit - 1)
        x = x.round()
        x = x / (2 ** output_bit - 1)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None
