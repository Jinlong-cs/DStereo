# Copyright (c) Horizon Robotics. All rights reserved.
import logging
from typing import Sequence

from hat.registry import OBJECT_REGISTRY
from .visualize import BaseVisualize

__all__ = ["ComposeVisualize"]

logger = logging.getLogger(__name__)


@OBJECT_REGISTRY.register
class ComposeVisualize(BaseVisualize):
    """
    ComposeVisualize Callback is used for compose visualize results.

    All results are first written on a picture and saved,
    and then drawn according to the needs of users.

    Args:
        callbacks: All of callbacks for visualize.
    """

    def __init__(
        self,
        callbacks: Sequence[BaseVisualize],
    ):
        for callback in callbacks:
            assert isinstance(callback, BaseVisualize)
        self.callbacks = callbacks
        super().__init__()

    def on_batch_end(self, global_step_id, batch, model_outs, **kwargs):
        for callback in self.callbacks:
            if callback.match_task_output:
                batch_output, match_output = callback.match_task_output(
                    batch, model_outs
                )
            else:
                batch_output, match_output = batch, model_outs

            global_step_id, batch_output, match_output = callback.visualize(
                global_step_id, batch_output, match_output
            )

    def __repr__(self):
        return "ComposeVisualize"
