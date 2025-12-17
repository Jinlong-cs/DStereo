from hat.callbacks.pupil_segmentation import LossWeightUpdater


class TestModel:
    def __init__(self):
        self.module = TestModule()


class TestModule:
    def __init__(self):
        self.losses = TestLoss()


class TestLoss:
    def __init__(self):
        self.loss_weights = {"num_epochs": 40}


def test_loss_weight_updater():
    weight_updater = LossWeightUpdater()
    model = TestModel()
    gt_weights = [0.0, 0.025, 0.05]
    for i in range(3):
        weight_updater.on_epoch_begin(model, i)
        new_w = model.module.losses.loss_weights["surface_loss_ratio"]
        assert new_w == gt_weights[i]
