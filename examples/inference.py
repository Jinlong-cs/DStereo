from examples.classification.resnet18 import *

# inference for local or aidi
preprocess = None
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
