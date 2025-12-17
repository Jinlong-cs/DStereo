import torch
from horizon_plugin_pytorch.march import March

# inference for aidipredictor


def camera_frame_list_to_dict(msgs):
    img_meta = dict(
        img=[],
    )

    for msg in msgs:
        img = msg.image.data
        img_meta["img"].append(img)
    img_meta["img"] = torch.stack(img_meta["img"])

    return img_meta


march = March.BAYES

deploy_model = dict(
    type="Classifier",
    backbone=dict(
        type="ResNet18",
        num_classes=1000,
        bn_kwargs={},
        flat_output=False,
    ),
    losses=None,
)

preprocess = [camera_frame_list_to_dict]
postprocess = None

inference = dict(
    type="Inference",
    device=None,
    march=march,
    model=deploy_model,
    pre_processors=preprocess,
    post_processors=postprocess,
    model_convert_pipeline=None,
)
