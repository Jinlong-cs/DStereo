# from hat.core.data_struct.workflow_struct import BBox3D
try:
    from hatbc.message import BBox3D
except ImportError:
    BBox3D = None

from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["MTFCOS3DOutputParser"]


@OBJECT_REGISTRY.register
class MTFCOS3DOutputParser:
    """MTFCOS3D parser used to standarize outputs from decoder outputs.

    Args:
        class_name: name of decoded class.
        use_2d_score: using 2d_score or 3d_score for final object's score.
        score_threshold: threshold for filtering object.
    """

    @require_packages("hatbc")
    def __init__(
        self,
        class_name: str,
        use_2d_score: bool = False,
        score_threshold: float = 0.4,
    ):

        self.class_name = class_name
        self.use_2d_score = use_2d_score
        self.score_threshold = score_threshold

    def __call__(self, output):
        if self.use_2d_score:
            score = output["score2d"]
        else:
            score = output["score"]
        keep = output["nms_keep"]
        filtered_mask = score > self.score_threshold
        keep_index = filtered_mask & keep.bool()
        for key, value in output.items():
            output[key] = value[keep_index]
        score = score[keep_index]
        results = []
        for ib in range(len(score)):
            results.append(
                BBox3D(
                    topic=self.class_name,
                    alpha=float(output["alpha"][ib].cpu()),
                    yaw=float(output["rotation_y"][ib].cpu()),
                    dim=output["dim"][ib].cpu().numpy(),
                    loc=output["location"][ib].cpu().numpy(),
                    score=float(score[ib].cpu()),
                )
            )
        return results
