#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import pytest
from torch import nn
from torch.nn.modules import Identity, ReLU, Threshold

from hat.models.losses.l1_loss import L1Loss
from hat.models.losses.mse_loss import MSELoss
from hat.utils.module_patch import (
    ModuleShareError,
    TorchModulePatch,
    merge_symbol_nodes,
)
from hat.utils.package_helper import check_packages_available

hatbc_available = check_packages_available("hatbc", raise_exception=False)


if hatbc_available:
    from hatbc.workflow.proxy import Variable, get_traced_graph
    from hatbc.workflow.trace import GraphTracer


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.thresh(out)
        return out


class ModelShare(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")
        self.act = Identity(node_name="act")

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.act(self.thresh(out))
        return out


class ModelShareErrArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.1, inplace=True, node_name="thresh")
        self.act = Identity()

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.act(self.thresh(out))
        return out


class ModelShareErrType(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = ReLU(inplace=True, node_name="thresh")
        self.act = Identity()

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.act(self.thresh(out))
        return out


class ModelShareErrKWArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.1, inplace=False, node_name="thresh")
        self.act = Identity()

    def forward(self, data, label):
        out = self.loss(data, label)
        out = self.act(self.thresh(out))
        return out


class DictLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1loss = L1Loss()
        self.l2loss = MSELoss()

    def forward(self, data, label):
        return {
            "l1loss": self.l1loss(data, label),
            "l2loss": self.l2loss(data, label),
        }


class ModelMergeWithArgs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DictLoss(node_name="loss")
        self.thresh = Threshold(0.5, 0.1, inplace=False, node_name="thresh")

    def forward(self, data, label):
        out = self.loss(data, label)["l1loss"]
        return self.thresh(out)


class ModelMergeWithArgs2(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = DictLoss(node_name="loss")
        self.thresh = Threshold(0.5, 0.1, inplace=False, node_name="thresh")

    def forward(self, data, label):
        out = self.loss(data, label)["l2loss"]
        return self.thresh(out)


class ToyModule_1(nn.Module):
    def forward(self, img):
        return img * 1.2


class ToyModule_2(nn.Module):
    def forward(self, data: dict):
        return data["feats"][0] * 1.2


class MergeWithDictArgs1(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = ToyModule_1(node_name="m1")
        self.m2 = ToyModule_2(node_name="m2")

    def forward(self, data: dict):
        x = self.m1(data["img"])
        feats = {"feats": [x]}
        output = self.m2(feats)

        return output


class MergeWithDictArgs2(nn.Module):
    def __init__(self):
        super().__init__()
        self.m1 = ToyModule_1(node_name="m1")
        self.m2 = ToyModule_2(node_name="m2")

    def forward(self, data: dict):
        x = self.m1(data["img"])
        feats = {"feats": [x]}
        output = self.m2(feats)

        return output


class MergeWithKWArgs(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")

    def forward(self, data, label):
        out = self.loss(data, label=label)
        out = self.thresh(out)
        return out


class MergeWithKWArgs2(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = L1Loss(node_name="loss")  # enable sharing
        self.thresh = Threshold(0.5, 0.5, inplace=True, node_name="thresh")

    def forward(self, data, label1, label2):
        out = self.loss(data, label=[label1, label2])
        out = self.thresh(out)
        return out


@pytest.mark.skipif(not hatbc_available, reason="need hatbc")
class TestTorchModulePatch:
    def setup(self):
        TorchModulePatch._workspaces.clear()

    def test_module_sharing_same_workspace(self):
        with TorchModulePatch():
            model1 = Model()
            model2 = ModelShare()

        assert id(model1.loss) == id(model2.loss)
        assert id(model1.thresh) == id(model2.thresh)

    def test_module_sharing_different_workspace(self):

        with TorchModulePatch(key="w1"):
            model1 = Model()
        with TorchModulePatch(key="w2"):
            model2 = ModelShare()

        assert id(model1.loss) != id(model2.loss)
        assert id(model1.thresh) != id(model2.thresh)

    def test_module_sharing_different_workspace_cascade(self):

        with TorchModulePatch(key="w1"):
            model1 = Model()
            with TorchModulePatch(key="w2"):
                model2 = ModelShare()

            assert id(model1.loss) != id(model2.loss)
            assert id(model1.thresh) != id(model2.thresh)

    def test_module_sharing_diff_op(self):
        with pytest.raises(ModuleShareError) as e:
            with TorchModulePatch():
                _ = Model()
                _ = ModelShareErrType()
        print(e)

    def test_module_sharing_diff_args(self):
        with pytest.raises(ModuleShareError) as e:
            with TorchModulePatch():
                _ = Model()
                _ = ModelShareErrArgs()
        print(e)

    def test_module_sharing_diff_kwargs(self):
        with pytest.raises(ModuleShareError) as e:
            with TorchModulePatch():
                _ = Model()
                _ = ModelShareErrKWArgs()
        print(e)

    def test_module_trace(self):
        with TorchModulePatch():
            data = Variable("data")
            label = Variable("label")
            model1 = Model()
            model2 = ModelShare()
            with GraphTracer(imperative=False):
                out1 = model1(data, label)
                out2 = model2(data, label)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name() == {"Threshold2", "L1Loss1"}

    def test_module_trace_disable_trace_init(self):
        with TorchModulePatch():
            data = Variable("data")
            label = Variable("label")
            model1 = Model(traceable=False)
            model2 = ModelShare(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data, label)
                out2 = model2(data, label)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "L1Loss2",
                    "Identity1",
                    "L1Loss1",
                    "data",
                    "Threshold2",
                    "Threshold1",
                    "label",
                }
                assert graph.get_children_name() == {"L1Loss1", "Threshold2"}

    def test_module_trace_disable_trace_forward(self):
        with TorchModulePatch():
            data = Variable("data")
            label = Variable("label")
            model1 = Model()
            model2 = ModelShare()
            with GraphTracer(imperative=False):
                out1 = model1(data, label, disable_trace=True)
                out2 = model2(data, label, disable_trace=True)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "L1Loss2",
                    "Identity1",
                    "L1Loss1",
                    "data",
                    "Threshold2",
                    "Threshold1",
                    "label",
                }
                assert graph.get_children_name() == {"L1Loss1", "Threshold2"}

    def test_module_trace_merge_graph(self):
        with TorchModulePatch():
            data = Variable("data")
            label = Variable("label")
            model1 = Model(traceable=False)
            model2 = ModelShare(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data, label)
                out2 = model2(data, label)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "L1Loss2",
                    "Identity1",
                    "L1Loss1",
                    "data",
                    "Threshold2",
                    "Threshold1",
                    "label",
                }
                merge_symbol_nodes(graph)
                assert graph.get_children_name(recursive=True) == {
                    "Identity1",
                    "L1Loss1",
                    "data",
                    "Threshold1",
                    "label",
                }

    def test_module_trace_merge_graph_with_args(self):
        """
        Test that node with same input but different args should not be merged
        """
        with TorchModulePatch():
            data = Variable("data")
            label = Variable("label")
            model1 = ModelMergeWithArgs1(traceable=False)
            model2 = ModelMergeWithArgs2(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data, label)
                out2 = model2(data, label)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "DictLoss2",
                    "DictLoss1",
                    "getitem1",
                    "getitem2",
                    "data",
                    "Threshold2",
                    "Threshold1",
                    "label",
                }
                merge_symbol_nodes(graph)
                assert graph.get_children_name(recursive=True) == {
                    "DictLoss1",
                    "getitem1",
                    "getitem2",
                    "data",
                    "Threshold2",
                    "Threshold1",
                    "label",
                }

    def test_module_trace_merge_graph_with_dict_args(self):
        """
        Test that node with same input but different args should not be merged
        """
        with TorchModulePatch():
            data = Variable("data")
            model1 = MergeWithDictArgs1(traceable=False)
            model2 = MergeWithDictArgs2(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data)
                out2 = model2(data)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "ToyModule_11",
                    "ToyModule_12",
                    "ToyModule_21",
                    "ToyModule_22",
                    "data",
                    "getitem1",
                    "getitem2",
                }
                merge_symbol_nodes(graph)
                print(graph.get_children_name(recursive=True))
                assert graph.get_children_name(recursive=True) == {
                    "ToyModule_11",
                    "ToyModule_21",
                    "data",
                    "getitem1",
                }

    def test_module_trace_merge_graph_with_kwargs(self):
        """
        Test that node with different kwargs should not be merged
        """
        with TorchModulePatch():
            data = Variable("data")
            label1 = Variable("label1")
            label2 = Variable("label2")
            model1 = MergeWithKWArgs(traceable=False)
            model2 = MergeWithKWArgs(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data, label1)
                out2 = model2(data, label2)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "Threshold1",
                    "data",
                    "L1Loss1",
                    "L1Loss2",
                    "label1",
                    "Threshold2",
                    "label2",
                }
                merge_symbol_nodes(graph)
                assert graph.get_children_name(recursive=True) == {
                    "Threshold1",
                    "data",
                    "L1Loss1",
                    "L1Loss2",
                    "label1",
                    "Threshold2",
                    "label2",
                }

    def test_module_trace_merge_graph_with_kwargs2(self):
        """
        Test that node with different kwargs should not be merged
        """
        with TorchModulePatch():
            data = Variable("data")
            label1 = Variable("label1")
            label2 = Variable("label2")
            model1 = MergeWithKWArgs2(traceable=False)
            model2 = MergeWithKWArgs2(traceable=False)
            with GraphTracer(imperative=False):
                out1 = model1(data, label1, label2)
                out2 = model2(data, label1, label2)
                graph = get_traced_graph([out1, out2])
                assert graph.get_children_name(recursive=True) == {
                    "Threshold1",
                    "data",
                    "L1Loss1",
                    "L1Loss2",
                    "label1",
                    "Threshold2",
                    "label2",
                }
                merge_symbol_nodes(graph)
                print(graph.get_children_name(recursive=True))
                assert graph.get_children_name(recursive=True) == {
                    "Threshold1",
                    "data",
                    "L1Loss1",
                    "label1",
                    "label2",
                }


if __name__ == "__main__":
    pytest.main(["-s", __file__])
