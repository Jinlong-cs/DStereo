try:
    import lpips
except ImportError:
    lpips = None

import torch

from hat.models.losses.face3d import Face3dLoss


def gen_fake_data():
    num_ldmk = 68
    num_verts = 5023
    batch_size = 7
    h, w = 128, 128
    data = {}
    data["pred"] = {
        "img_ldmk": torch.randn((batch_size, num_ldmk, 2)),
        "cam_ldmk": torch.randn((batch_size, num_ldmk, 3)),
        "cam_verts": torch.randn((batch_size, num_verts, 3)),
        "real_verts": torch.randn((batch_size, num_verts, 3)),
        "global_pose": torch.randn((batch_size, 3)),
        "transl": torch.randn((batch_size, 3)),
        "jaw": torch.randn((batch_size, 3)),
        "exp": torch.randn((batch_size, 50)),
        "shape": torch.randn((batch_size, 100)),
        "light": torch.randn((batch_size, 3, 9)),
        "tex": torch.randn((batch_size, 50)),
        "pr_img": torch.randn((batch_size, 3, h, w)),
        "pr_depth": torch.randn((batch_size, 1, h, w)),
        "pr_mask": torch.ones((batch_size, 1, h, w)),
        "real_eye3d_left": torch.randn((batch_size, 3)),
        "real_eye3d_right": torch.randn((batch_size, 3)),
    }
    data["label"] = {
        "gt_ldmk": torch.randn((batch_size, num_ldmk, 2)),
        "gt_cam_ldmk": torch.randn((batch_size, num_ldmk, 3)),
        "gt_verts": torch.randn((batch_size, num_verts, 3)),
        "global_pose": torch.randn((batch_size, 3)),
        "transl": torch.randn((batch_size, 3)),
        "jaw": torch.randn((batch_size, 3)),
        "exp": torch.randn((batch_size, 50)),
        "shape": torch.randn((batch_size, 100)),
        "light": torch.randn((batch_size, 3, 9)),
        "tex": torch.randn((batch_size, 50)),
        "gt_img": torch.randn((batch_size, 3, h, w)),
        "gt_depth": torch.randn((batch_size, 1, h, w)),
        "eye3d_left": torch.randn((batch_size, 3)),
        "eye3d_right": torch.randn((batch_size, 3)),
    }
    loss_weights = {
        "img_ldmk": 1,  # 256.0,
        "cam_ldmk": 1,
        "cam_verts": 1,
        "photo": 1,  # 10.0,
        "lpips": 1,  # 30.0,
        "eye3d": 1,
        "depth": 1,
        "global_pose": 1,
        "transl": 1,
        "shape": 1,
        "jaw": 1,
        "exp": 1,
        "tex": 1,
        "light": 1,
        "shape_cons": 0,
        "shape_reg": 5e-2,
        "exp_reg": 1e-2,
        "tex_reg": 1,
        "eyelid": 0,
    }
    return data, loss_weights


def test_face3d_loss():
    data, loss_weights = gen_fake_data()
    if lpips is None:
        return
    lpips_module = lpips.LPIPS(
        pretrained=True,
        pnet_rand=True,
        model_path="tmp_orig_data/face/face3d/hat_test/face3d_params/lpips.pth",  # noqa
        verbose=True,
        net="vgg",
    )
    face3d_loss = Face3dLoss(loss_weights, lpips_module, 1)
    loss = face3d_loss(data)
    for _, v in loss.items():
        assert v >= 0
