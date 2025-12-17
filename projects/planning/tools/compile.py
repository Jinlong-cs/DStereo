import os

import torch
from horizon_plugin_pytorch import quantization

quantization.march = quantization.March.BAYES
torch.backends.cudnn.deterministic = True

hbdk_dir = "your_dir"
model_path = os.path.join(hbdk_dir, "deploy-checkpoint-last.pt")

traced_model = torch.jit.load(model_path)
example_input = dict(
    road_map=torch.zeros((1, 1, 512, 512)),
    rendered_obs=torch.zeros((1, 4, 512, 512)),
    plan_svf_goal=torch.zeros((1, 1, 65, 65)),
    plan_ego_motion=torch.ones(size=(1, 2, 65, 65)),
)


# compile and perf
quantization.check_model(
    traced_model, example_input, march=quantization.march, advice=1
)

quantization.compile_model(
    traced_model,
    example_input,
    opt=2,
    march=quantization.march,
    name="plan_imitation_value",
    hbm=hbdk_dir + "/int_model_add_desc.hbm",
)

quantization.perf_model(
    traced_model.eval(),
    example_input,
    opt=2,
    march=quantization.march,
    input_source=["ddr"],
    layer_details=True,
    out_dir=hbdk_dir,
)
