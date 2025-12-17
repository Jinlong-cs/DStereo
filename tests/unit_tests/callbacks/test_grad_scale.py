import copy

import pytest
import torch
from torch.cuda.amp import GradScaler, autocast

from hat.callbacks.grad_scale import GradScale


@pytest.mark.parametrize("enable_amp", [False, True])
def test_grad_scale(enable_amp):
    model = torch.nn.Linear(2, 3)
    torch.nn.init.zeros_(model.weight)
    torch.nn.init.zeros_(model.bias)
    model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), 0.01)
    batch = torch.ones(1, 2, requires_grad=True).cuda()

    # build callback
    callback = GradScale(
        module_and_scale=[
            ("weight", 0.1, None),
            ("bias", 10, None),
        ],
        clip_grad_norm=None,
    )

    callback.on_loop_begin()
    auto_cast = autocast(enabled=enable_amp, dtype=torch.float16)
    grad_scaler = GradScaler(enabled=enable_amp)
    with auto_cast:
        output = model(batch)

    grad_scaler.scale(output).backward(
        torch.ones(1, 3).cuda() * 2, retain_graph=True
    )
    before_callback_w_grad = copy.deepcopy(model.weight.grad.detach())
    before_callback_b_grad = copy.deepcopy(model.bias.grad.detach())

    callback.on_backward_end(model, batch=batch, optimizer=optimizer)
    assert torch.sum(model.weight.grad) == 0
    assert torch.sum(model.bias.grad) == 0

    callback.on_optimizer_step_begin(
        model, optimizer=optimizer, grad_scaler=grad_scaler
    )
    assert torch.all((before_callback_w_grad * 0.1) == model.weight.grad)
    assert torch.all((before_callback_b_grad * 10) == model.bias.grad)
    optimizer.step()
