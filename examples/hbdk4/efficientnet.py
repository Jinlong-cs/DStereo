from examples.classification.efficientnet import *


class ClassifierHbirWrapper:
    def __init__(self, save_path, batch_transforms=None) -> None:
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        import torchvision

        from hat.registry import build_from_registry

        if batch_transforms is not None:
            batch_transforms = build_from_registry(batch_transforms)
            if isinstance(batch_transforms, (list, tuple)):
                batch_transforms = torchvision.transforms.Compose(
                    batch_transforms
                )  # noqa
        self.batch_transforms = batch_transforms

        self.step_count = 0
        self.input = None
        self.target = None

        from hat.metrics.acc import Accuracy

        self.acc = Accuracy()

    def pre_process(self, batch):
        batch = self.batch_transforms(batch)
        image = batch["img"]
        self.input = image
        self.target = batch.get("labels", None)
        return image

    def post_process(self, preds):
        import numpy as np

        self.acc.update(self.target, preds[0])
        if self.save_path is not None:
            save_name = "hbir_inout_{}.npy".format(self.step_count)
            np.save(
                os.path.join(self.save_path, save_name),
                {"input": self.input.numpy(), "output": preds[0]},
            )

        self.step_count += 1

    def end_process(self):
        results = self.acc.get_name_value()
        for name, value in results:
            print("{}: {}".format(name, value))


hbir_wrapper = ClassifierHbirWrapper(
    os.path.join(ckpt_dir, "hbir_inout"),
    batch_transforms=batch_processor["batch_transforms"],
)
hbir_dataloader = copy.deepcopy(val_data_loader)
hbir_dataloader["batch_size"] = 1

hbir_predictor = dict(
    type="HbirPredictor",
    model_path=os.path.join(ckpt_dir, "quantized.bc"),
    dataloader=hbir_dataloader,
    pre_process=hbir_wrapper.pre_process,
    post_process=hbir_wrapper.post_process,
    end_process=hbir_wrapper.end_process,
)
