import contextlib
import logging

try:
    from matrix_gluon.data.dataset.common import DefaultBufWithAnnoDataset
except ImportError:
    DefaultBufWithAnnoDataset = object


from hat.registry import OBJECT_REGISTRY
from hat.utils.deprecate import deprecated_warning
from hat.utils.logger import OutputLogger
from hat.utils.package_helper import require_packages

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class LegacyMXDefaultBufWithAnnoDataset(DefaultBufWithAnnoDataset):
    @require_packages("matrix_gluon")
    def __init__(
        self,
        img_rec_path,
        anno_rec_path,
        decode_img=False,
        decode_anno=False,
        to_rgb=False,
        transforms=None,
    ):
        deprecated_warning(
            author="tian.li",
            old_name="LegacyMXDefaultBufWithAnnoDataset",
            deprecation_version="v1.2.0",
            removal_version="v1.2.2",
        )

        with contextlib.redirect_stdout(OutputLogger(logger)):
            super().__init__(
                img_rec_path, anno_rec_path, decode_img, decode_anno, to_rgb
            )
        self.transforms = transforms

    def __getitem__(self, idx):
        img, anno = super().__getitem__(idx)
        data = {
            "img": img,
            "anno": anno,
        }
        if self.transforms is not None:
            for transform in self.transforms:
                data = transform(data)
        return data
