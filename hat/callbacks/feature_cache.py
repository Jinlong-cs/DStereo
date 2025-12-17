# Copyright (c) Horizon Robotics. All rights reserved.
import logging
import sys
from typing import Mapping, Optional

from hatbc.workflow.symbol import Node
from hatbc.workflow.visualize import visualize_graph

from hat.registry import OBJECT_REGISTRY
from hat.utils.cache import get_global_cache
from hat.utils.model_helpers import get_binding_module
from .callbacks import CallbackMixin

__all__ = ["FeatureCache"]

logger = logging.getLogger(__name__)


# This name is auto defined by hatbc'workflow.
# `cache_feature_op` is upper operator name, and `1` is auto added by hatbc.
CACHE_FEATURE_NODE_NAME = "cache_feature_op1"


def update_graph_for_feature_cache(
    graph_model,
    wrapped_input_name: str,
    vis_graph: Optional[bool] = False,
) -> None:
    """Update graph model.

    Args:
        graph_model: input graph model. This operation is done inplace.
        wrapped_input_name: new input name for sub-graph model.
        vis_graph: save graph to png for debug.

    """
    from hat.models.structures.graph_model import GraphModel

    assert isinstance(
        graph_model, GraphModel
    ), "feature cache only support GraphyModel!"
    g = graph_model.graph

    if vis_graph:
        visualize_graph(g, save_path="tmp_graph_raw.png")

    # 1. find all node after `cache_feature_op` node
    nodes_after_feature_cache = []
    found_cache_feature_node = False

    def fvisit_after(node):
        nonlocal found_cache_feature_node
        for input_node in node.inputs:
            print(input_node.name)
            if CACHE_FEATURE_NODE_NAME in input_node.name:
                nodes_after_feature_cache.append(node)
                assert input_node.name == CACHE_FEATURE_NODE_NAME
                found_cache_feature_node = True

    g.post_order_dfs_visit(fvisit_after)
    assert found_cache_feature_node, (
        f"{CACHE_FEATURE_NODE_NAME} not found in" "graph, please check graph."
    )

    # 2. replace `img` with `cached_feature` in input_nodes
    cache_feature_node = Node.create_placeholder(name=wrapped_input_name)
    for node in nodes_after_feature_cache:
        # update node _inputs
        new_inputs = {
            n: n for n in node._inputs if CACHE_FEATURE_NODE_NAME not in n.name
        }
        node._inputs = new_inputs
        node._inputs.update({cache_feature_node: cache_feature_node})
        # update node _args
        cache_feature1_found = False
        new_args = []
        for _arg in node._args:
            if (
                hasattr(_arg, "name") and _arg.name == CACHE_FEATURE_NODE_NAME
            ):  # noqa
                new_args.append(cache_feature_node)
                cache_feature1_found = True
            else:
                new_args.append(_arg)
        node._args = tuple(new_args)
        assert cache_feature1_found, (
            f"{CACHE_FEATURE_NODE_NAME} not " "found in graph model"
        )

    # 3. add `wrapped_input_name` to graph_model's inputs
    graph_model.inputs[wrapped_input_name] = None

    if vis_graph:
        visualize_graph(g, save_path="tmp_graph_updated.png")


@OBJECT_REGISTRY.register
class FeatureCache(CallbackMixin):
    """Feature cache callback.

    This callback does two things in two hooks:

        1) In "write_cache" mode, it closes `global_cache` handle and exit
           program if `exit_when_write_finish` is True.
           This is done in `on_epoch_end`.
        2) In "read_cache" mode, it updates graph model and remove useless
           node in graph model to speed up training. It's done in
           `on_epoch_begin` now, maybe it should be moved to `on_loop_begin`.

    Args:
        cache_file: Feature cache file path.
        wrapped_input_name: New input name for new graph model.
            When using feature cache, graph model will be updated, leaving
            sub-graph for fast training. `wrapped_input_name` is used as
            new feature input placeholder for sub-graph.
            Default is "cached_feature".
            Note: it should be same as CacheDataset's input_name in
            `hat/data/datasets/cache_dataset.py`.
        read_cache: Set to `read_cache` mode. Default is False.
        write_cache: Set to `write_cache` mode. Default is False.
            Note: `read_cache` and `write_cache` can't be False together.
        exit_when_write_finish: Exit program when write finish.
            Default is True.
        vis_graph: Visualize graph model. Used for debug.
            Default is False.
        batch_length: a dictionary specify the length in one batch
            of each data (along dim 0).

    """

    def __init__(
        self,
        cache_file: str,
        wrapped_input_name: Optional[str] = "cached_feature",
        read_cache: Optional[bool] = False,
        write_cache: Optional[bool] = False,
        exit_when_write_finish: Optional[bool] = True,
        vis_graph: Optional[bool] = False,
        batch_length: Optional[Mapping] = None,
    ):
        self.cache_file = cache_file
        self.wrapped_input_name = wrapped_input_name
        self.vis_graph = vis_graph
        self.read_cache = read_cache
        self.write_cache = write_cache
        self.batch_length = batch_length
        # as least one is True
        assert read_cache + write_cache == 1
        self.exit_when_write_finish = exit_when_write_finish

    def on_epoch_begin(self, epoch_id, model, data_loader, **kwargs):
        """Update graph model on epoch begin."""
        if epoch_id == 0 and self.read_cache:
            model = get_binding_module(model)
            update_graph_for_feature_cache(
                model,
                wrapped_input_name=self.wrapped_input_name,
                vis_graph=self.vis_graph,
            )
            logger.info("update graph model for feature cache")

    def on_epoch_end(self, epoch_id, **kwargs):
        """If in `write_cache` mode, close cache and exit program if set."""
        if epoch_id == 0 and self.write_cache:
            global_cache = get_global_cache(
                self.cache_file, writable=False, batch_length=self.batch_length
            )
            global_cache.close()
            logger.info("writing cache finished")
            if self.exit_when_write_finish:
                logger.info("exit")
                sys.exit()
