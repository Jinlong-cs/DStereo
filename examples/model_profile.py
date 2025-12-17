from examples.classification.resnet18 import *

model_profiler_solver = dict(
    model=deploy_model,
    inputs=deploy_inputs,
    model_convert_pipeline=[
        float_predictor["model_convert_pipeline"],
        calibration_predictor["model_convert_pipeline"],
    ],
    tool=dict(
        type="FeaturemapSimilarity",
        similarity_func="Cosine",
        threshold=None,
    ),
)
