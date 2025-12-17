import os
import pickle

import numpy as np
from hpflow.eval_platform_adaptor import EvalPlatformAdaptor
from hpflow.structure.ptype import (
    PTypeEnum,
    ptype2eval_platform_type,
    ptype2pe_to_4pe,
)
from hpflow.utils.utils import TakeByKey

PERSON_6_CLASS = os.getenv("HPFLOW_PERSON_6_CLASS", "0") == "1"


class TakePredictionByKey(TakeByKey):
    def __call__(self, x):
        if x is None:
            return None
        assert isinstance(x, dict)
        outputs = [
            x[ptype2pe_to_4pe[key_i]] if key_i in ptype2pe_to_4pe else x[key_i]
            for key_i in self.keys
        ]
        if self.as_list:
            return outputs
        else:
            return outputs[0] if len(outputs) == 1 else outputs


class PersonAdapter(EvalPlatformAdaptor):
    def __init__(
        self,
        key,
        take_input_func=None,
        person_head_clip_thres=0.7,
        nms_thres=0.5,
        joint_4pe_2pe=True,
        score_thresh_4pe=0.35,
        score_thrush_2pe=0.1,
    ):
        self.take_input_func = take_input_func
        self.person_head_clip_thres = person_head_clip_thres
        self.key = key
        self.nms_thres = nms_thres
        self.joint_4pe_2pe = joint_4pe_2pe
        self.score_thresh_4pe = score_thresh_4pe
        self.score_thrush_2pe = score_thrush_2pe

    @staticmethod
    def softmax_output(prob):
        softmax = np.exp(prob - np.max(prob, keepdims=True))
        softmax = softmax / np.sum(softmax, keepdims=True)
        return softmax

    def __call__(self, perceptions):
        perceptions, image_name = perceptions
        if isinstance(perceptions, bytes):
            perceptions = pickle.loads(perceptions)
        assert isinstance(self.key, str)

        if self.take_input_func is not None:
            perceptions = self.take_input_func(perceptions)
        # init objects
        if self.key in ptype2pe_to_4pe and self.key not in [
            PTypeEnum.kPedHeadBBox2D,
            PTypeEnum.kPedPosNeg,
        ]:
            eval_key = "objects"
            objects = dict({eval_key: list()})
        else:
            eval_key = ptype2eval_platform_type[self.key]
            objects = dict({eval_key: list()})
        if perceptions is None:
            return objects

        for obj_idx, perception in enumerate(perceptions):
            if not perception or not perception.is_valid:
                continue
            # convert 4PE results to label platform
            if self.key not in ptype2pe_to_4pe:
                self.add_4pe_instance(eval_key, objects, perception)
            # convert 2PE results to label platform
            # ## for 2pe tasks that do not need extra convert
            if self.key in [
                PTypeEnum.kPedAge,
                PTypeEnum.kPedPose,
                PTypeEnum.kPedOrientation,
            ]:
                self.add_2pe_attr_common(
                    eval_key, objects, obj_idx, perception
                )  # noqa
            # ## convert person head
            elif self.key in [PTypeEnum.kPedHeadBBox2D]:
                self.add_2pe_person_head(eval_key, objects, perception)
            elif self.key in [PTypeEnum.kPedPosNeg] and self.joint_4pe_2pe:
                self.add_joint_4pe_2pe(eval_key, objects, perception)
            elif self.key in [
                PTypeEnum.kPedPosNeg,
                PTypeEnum.kPedOcclusion,
                PTypeEnum.kPedPosNegOcclusion,
            ]:
                self.add_2pe_posneg_occ(
                    eval_key, objects, obj_idx, perception
                )  # noqa

        # apply extra clean ups to objects
        if self.key in [PTypeEnum.kPedHeadBBox2D]:
            dets = np.array(
                [
                    [
                        i["bbox"][0],
                        i["bbox"][1],
                        i["bbox"][2],
                        i["bbox"][3],
                        i["bbox_score"],
                    ]
                    for i in objects[eval_key]
                ]
            )
            if len(dets) > 0:
                keeps = self.apply_nms(dets, thresh=self.nms_thres)
                objects[eval_key] = [objects[eval_key][i] for i in keeps]
        return objects

    def add_4pe_instance(self, eval_key, objects, perception):
        assert self.key in [PTypeEnum.kPedBBox2D, PTypeEnum.kCycBBox2D]
        objects[eval_key].append(
            dict(
                bbox=[
                    float(perception.x1),
                    float(perception.y1),
                    float(perception.x2),
                    float(perception.y2),
                ],
                bbox_score=float(perception.conf),
                attrs=dict(),
            )
        )

    def add_2pe_attr_common(self, eval_key, objects, obj_idx, perception):
        class2num_cls = {
            PTypeEnum.kPedAge: 2,
            PTypeEnum.kPedPose: 5,
            PTypeEnum.kPedOrientation: 8,
        }
        task_prefix = ptype2eval_platform_type[self.key]
        p = perception.attributes.get(self.key)
        if p is None:
            objects[eval_key].append(
                {
                    "obj_id": obj_idx,
                    f"{task_prefix}_prediction": 0,
                    f"{task_prefix}_scores": [0.0] * class2num_cls[self.key],
                    "attrs": {"ignore": "yes"},
                }
            )
        else:
            objects[eval_key].append(
                {
                    "obj_id": obj_idx,
                    f"{task_prefix}_prediction": p.scores.index(
                        max(p.scores)
                    ),  # noqa
                    f"{task_prefix}_scores": p.scores,
                }
            )

    def add_2pe_person_head(self, eval_key, objects, perception):
        def _check_valid():
            box_4pe = [
                float(perception.x1),
                float(perception.y1),
                float(perception.x2),
                float(perception.y2),
            ]
            x1_max = max(box_4pe[0], bbox[0])
            y1_max = max(box_4pe[1], bbox[1])
            x2_min = min(box_4pe[2], bbox[2])
            y2_min = min(box_4pe[3], bbox[3])
            if (y2_min - y1_max) <= 0 or (x2_min - x1_max) <= 0:
                return False
            inter = (y2_min - y1_max) * (x2_min - x1_max)
            s2 = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            ioa = inter / s2
            return ioa > self.person_head_clip_thres

        p_heads = perception.attributes.get(self.key)

        for p_head in p_heads:
            bbox = [
                float(p_head.x1),
                float(p_head.y1),
                float(p_head.x2),
                float(p_head.y2),
            ]
            if _check_valid() and p_head.conf > self.score_thrush_2pe:
                objects[eval_key].append(
                    dict(
                        bbox=bbox, bbox_score=float(p_head.conf), attrs=dict()
                    )
                )

    def add_joint_4pe_2pe(self, eval_key, objects, perception):
        if PERSON_6_CLASS:
            p = perception.attributes.get(self.key)
        else:
            p = perception.attributes.get(PTypeEnum.kPedOcclusion)
        if p is None:
            return None
        score = p.scores
        if perception.conf <= self.score_thresh_4pe:
            return None
        if self.key == PTypeEnum.kPedPosNeg:
            score_fp = self.softmax_output(score)
            # for fp
            score_fp = [score_fp[0], sum(score_fp[1:])]
            if score_fp[0] > score_fp[1]:
                return None
            # softmax
            objects[eval_key].append(
                dict(
                    bbox=[
                        float(perception.x1),
                        float(perception.y1),
                        float(perception.x2),
                        float(perception.y2),
                    ],
                    bbox_score=perception.conf,
                    attrs=dict(),
                )
            )

    def add_2pe_posneg_occ(self, eval_key, objects, obj_idx, perception):
        if PERSON_6_CLASS:
            p = perception.attributes.get(self.key)
        else:
            # kPedPosNegOcclusion for 5 task composed
            p = perception.attributes.get(PTypeEnum.kPedPosNegOcclusion)
        if p is None:
            return None
        score = p.scores
        if self.key == PTypeEnum.kPedPosNeg:
            score_fp = self.softmax_output(score)
            # for fp
            score_fp = [score_fp[0], sum(score_fp[1:])]
            # softmax
            objects[eval_key].append(
                dict(
                    bbox=[
                        float(perception.x1),
                        float(perception.y1),
                        float(perception.x2),
                        float(perception.y2),
                    ],
                    bbox_score=score_fp[1],
                    attrs=dict(),
                )
            )
        elif self.key == PTypeEnum.kPedOcclusion:
            task_prefix = ptype2eval_platform_type[self.key]
            # for occlusion
            if PERSON_6_CLASS:
                score_occ = score
            else:
                score_occ = score[1:-1]
            objects[eval_key].append(
                {
                    "obj_id": obj_idx,
                    f"{task_prefix}_prediction": score_occ.index(
                        max(score_occ)
                    ),  # noqa
                    f"{task_prefix}_scores": score_occ,
                }
            )

    @staticmethod
    def apply_nms(dets, thresh):
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep
