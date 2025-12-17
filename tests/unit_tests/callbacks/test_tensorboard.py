import torch

from hat.registry import build_from_registry


def test_tensorboard(tmpdir):
    def tb_update_func_by_step(writer, model_outs, global_step_id):
        img = model_outs["img"]
        writer.add_image(
            tag="img",
            img_tensor=img,
            global_step=global_step_id,
            dataformats="NHWC",
        )

    cfg = dict(
        type="TensorBoard",
        save_dir=tmpdir,
        update_freq=2,
        tb_update_funcs=[tb_update_func_by_step],
    )
    tensorboard_by_step = build_from_registry(cfg)

    def tb_update_func_by_epoch(writer, epoch_id):
        writer.add_scalar(
            tag="epoch_id",
            scalar_value=epoch_id,
            global_step=epoch_id,
        )

    cfg = dict(
        type="TensorBoard",
        save_dir=tmpdir,
        update_freq=2,
        update_by="epoch",
        tb_update_funcs=[tb_update_func_by_epoch],
    )
    tensorboard_by_epoch = build_from_registry(cfg)

    fake_model_outs = dict(
        loss1=torch.tensor([1]),
        img=torch.randint(low=0, high=255, size=(1, 224, 224, 3)),
    )
    for epoch_id in range(2):
        for step_id in range(10):
            tensorboard_by_step.on_batch_end(
                model_outs=fake_model_outs, global_step_id=step_id
            )
            tensorboard_by_step.on_epoch_end(
                epoch_id=epoch_id, bad_param="bad param"
            )
            tensorboard_by_epoch.on_batch_end(
                model_outs=fake_model_outs,
                global_step_id=step_id,
                bad_param="bad param",
            )
            tensorboard_by_epoch.on_epoch_end(epoch_id=epoch_id)
