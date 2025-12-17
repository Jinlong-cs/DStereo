import torch

try:
    import kornia  # noqa

    KORNIA = True
except ImportError:
    KORNIA = False
from hat.models.task_modules.human3d.utils import FitsDict


def test_fits_dict():
    if not KORNIA:
        return

    fits_dict = FitsDict(
        load_dir="./tmp_orig_data/human3d/static_fits",
        checkpoint_dir="./tmp_models",
        dataset_list=["S311_gesture_cockpit_trainset_01"],
        age="adult",
    )

    opt_pose, opt_betas, _ = fits_dict[
        (
            ["S311_gesture_cockpit_trainset_01"],
            [10],
            torch.tensor([15.0]),
            torch.tensor([True]),
        )
    ]

    assert opt_pose.size() == (1, 72)
    assert opt_betas.size() == (1, 10)
