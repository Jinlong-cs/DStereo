import copy
import datetime
import json
import logging
import os
import sys
import time
from collections import defaultdict
from os import path as osp

import numpy as np
import torch.distributed as dist

try:
    from pycocotools import mask as coco_mask
    from pycocotools.coco import COCO
except ImportError:
    coco_mask = None
    COCO = None

from hat.metrics.metric import EvalMetric
from hat.registry import OBJECT_REGISTRY
from hat.utils.package_helper import require_packages

__all__ = ["COCOLdmkMetric"]

logger = logging.getLogger(__name__)


class HobotCOCOeval:
    """Interface for evaluating detection on the Microsoft COCO dataset.

    This class has been modified to fit different types of keypoints like
    hobot_kps17, hobot_kps19 and cockpit_kps15. For a detailed description
    of this class, refer to the official COCOeval class.

    Args:
        coco_gt: Coco object with ground truth annotations. Defaults to None.
        coco_dt: Coco object with detection results. Defaults to None.
        iou_type: Iou type. Defaults to "segm".
            Support type: "segm" and "bbox".
        ldmk_type: ldmk type. Defaults to "cockpit_kps15".
            Support type: "hobot_kps17", "hobot_kps19" and "cockpit_kps15".
    """

    @require_packages("pycocotools")
    def __init__(
        self,
        coco_gt=None,
        coco_dt=None,
        iou_type="segm",
        ldmk_type="cockpit_kps15",
    ):
        if not iou_type:
            logging.info("iou_type not specified. use default iou_type segm")
        self.coco_gt = coco_gt  # ground truth COCO API
        self.coco_dt = coco_dt  # detections COCO API
        self.eval_imgs = defaultdict(
            list
        )  # per-image per-category evaluation results [KxAxI] elements  # noqa
        self.eval = {}  # accumulated evaluation results
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        self.params = Params(iou_type=iou_type, ldmk_type=ldmk_type)
        if iou_type == "keypoints":
            self.ldmk_type = ldmk_type
            logging.info("ldmk_type = {}".format(ldmk_type))
        self._params_eval = {}  # parameters for evaluation
        self.stats = []  # result summarization
        self.ious = {}  # ious between all gts and dts
        if coco_gt is not None:
            self.params.img_ids = sorted(coco_gt.getImgIds())
            self.params.cat_ids = sorted(coco_gt.getCatIds())

    def _prepare(self):
        """Prepare ._gts and ._dts for evaluation based on params."""

        def _toMask(anns, coco):
            # modify ann['segmentation'] by reference
            for ann in anns:
                rle = coco.annToRLE(ann)
                ann["segmentation"] = rle

        p = self.params
        if p.use_cats:
            gts = self.coco_gt.loadAnns(
                self.coco_gt.getAnnIds(imgIds=p.img_ids, catIds=p.cat_ids)
            )
            dts = self.coco_dt.loadAnns(
                self.coco_dt.getAnnIds(imgIds=p.img_ids, catIds=p.cat_ids)
            )
        else:
            gts = self.coco_gt.loadAnns(
                self.coco_gt.getAnnIds(imgIds=p.img_ids)
            )
            dts = self.coco_dt.loadAnns(
                self.coco_dt.getAnnIds(imgIds=p.img_ids)
            )

        # convert ground truth to mask if iou_type == 'segm'
        if p.iou_type == "segm":
            _toMask(gts, self.coco_gt)
            _toMask(dts, self.coco_dt)
        # set ignore flag
        for gt in gts:
            gt["ignore"] = gt["ignore"] if "ignore" in gt else 0
            gt["ignore"] = "iscrowd" in gt and gt["iscrowd"]
            if p.iou_type == "keypoints":
                gt["ignore"] = (gt["num_keypoints"] == 0) or gt["ignore"]
        self._gts = defaultdict(list)  # gt for evaluation
        self._dts = defaultdict(list)  # dt for evaluation
        for gt in gts:
            self._gts[gt["image_id"], gt["category_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"], dt["category_id"]].append(dt)
        self.eval_imgs = defaultdict(
            list
        )  # per-image per-category evaluation results  # noqa
        self.eval = {}  # accumulated evaluation results

    def evaluate(self, check_scores=False):
        """Run per image evaluation on given images and store results (a list of dict) in self.eval_imgs."""  # noqa
        tic = time.time()
        logging.info("Running per image evaluation...")
        p = self.params
        # add backward compatibility if use_segm is specified in params
        if p.use_segm is not None:
            p.iou_type = "segm" if p.use_segm == 1 else "bbox"
            logging.info(
                "use_segm (deprecated) is not None. \
                Running {} evaluation".format(
                    p.iou_type
                )
            )
        logging.info("Evaluate annotation type *{}*".format(p.iou_type))
        p.img_ids = list(np.unique(p.img_ids))
        if p.use_cats:
            p.cat_ids = list(np.unique(p.cat_ids))
        p.max_dets = sorted(p.max_dets)
        self.params = p

        self._prepare()
        # loop through images, area range, max detection number
        cat_ids = p.cat_ids if p.use_cats else [-1]

        if p.iou_type == "segm" or p.iou_type == "bbox":
            compute_iou = self.compute_iou
        elif p.iou_type == "keypoints":
            compute_iou = self.compute_oks
        self.ious = {
            (img_id, cat_id): compute_iou(img_id, cat_id)
            for img_id in p.img_ids
            for cat_id in cat_ids
        }

        evaluate_img = self.evaluate_img
        max_det = p.max_dets[-1]
        self.eval_imgs = [
            evaluate_img(img_id, cat_id, area_rng, max_det, check_scores)
            for cat_id in cat_ids
            for area_rng in p.area_rng
            for img_id in p.img_ids
        ]
        self._params_eval = copy.deepcopy(self.params)
        toc = time.time()
        logging.info("DONE (t={:0.2f}s).".format(toc - tic))

    def compute_iou(self, img_id, cat_id):
        p = self.params
        if p.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [_ for cId in p.cat_ids for _ in self._gts[img_id, cId]]
            dt = [_ for cId in p.cat_ids for _ in self._dts[img_id, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return []
        inds = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in inds]
        if len(dt) > p.max_dets[-1]:
            dt = dt[0 : p.max_dets[-1]]

        if p.iou_type == "segm":
            g = [g["segmentation"] for g in gt]
            d = [d["segmentation"] for d in dt]
        elif p.iou_type == "bbox":
            g = [g["bbox"] for g in gt]
            d = [d["bbox"] for d in dt]
        else:
            raise Exception("unknown iou_type for iou computation")

        # compute iou between each dt and gt region
        iscrowd = [int(o["iscrowd"]) for o in gt]
        ious = coco_mask.iou(d, g, iscrowd)
        return ious

    def compute_oks(self, img_id, cat_id):
        p = self.params
        # dimention here should be Nxm
        gts = self._gts[img_id, cat_id]
        dts = self._dts[img_id, cat_id]
        inds = np.argsort([-d["score"] for d in dts], kind="mergesort")
        dts = [dts[i] for i in inds]
        if len(dts) > p.max_dets[-1]:
            dts = dts[0 : p.max_dets[-1]]
        # if len(gts) == 0 and len(dts) == 0:
        if len(gts) == 0 or len(dts) == 0:
            return []
        ious = np.zeros((len(dts), len(gts)))
        sigmas = p.kpt_oks_sigmas
        vars = (sigmas * 2) ** 2
        k = len(sigmas)
        # compute oks between each detection and ground truth object
        for j, gt in enumerate(gts):
            # create bounds for ignore regions(double the gt bbox)
            g = np.array(gt["keypoints"])
            xg = g[0::3]
            yg = g[1::3]
            vg = g[2::3]
            k1 = np.count_nonzero(vg > 0)
            bb = gt["bbox"]
            x0 = bb[0] - bb[2]
            x1 = bb[0] + bb[2] * 2
            y0 = bb[1] - bb[3]
            y1 = bb[1] + bb[3] * 2
            for i, dt in enumerate(dts):
                d = np.array(dt["keypoints"])
                xd = d[0::3]
                yd = d[1::3]
                if k1 > 0:
                    # measure the per-keypoint distance if keypoints visible
                    dx = xd - xg
                    dy = yd - yg
                else:
                    # measure minimum distance to keypoints in (x0,y0) & (x1,y1)  # noqa
                    z = np.zeros((k))
                    dx = np.max((z, x0 - xd), axis=0) + np.max(
                        (z, xd - x1), axis=0
                    )
                    dy = np.max((z, y0 - yd), axis=0) + np.max(
                        (z, yd - y1), axis=0
                    )
                e = (
                    (dx ** 2 + dy ** 2)
                    / vars
                    / (gt["area"] + np.spacing(1))
                    / 2
                )
                if k1 > 0:
                    e = e[vg > 0]
                ious[i, j] = np.sum(np.exp(-e)) / e.shape[0]
        return ious

    def evaluate_img(self, img_id, cat_id, arng, max_det, check_scores):
        """Perform evaluation for single category and image."""
        p = self.params
        if p.use_cats:
            gt = self._gts[img_id, cat_id]
            dt = self._dts[img_id, cat_id]
        else:
            gt = [_ for cId in p.cat_ids for _ in self._gts[img_id, cId]]
            dt = [_ for cId in p.cat_ids for _ in self._dts[img_id, cId]]
        if len(gt) == 0 and len(dt) == 0:
            return None

        for g in gt:
            if g["ignore"] or (g["area"] < arng[0] or g["area"] > arng[1]):
                g["_ignore"] = 1
            else:
                g["_ignore"] = 0

        # sort dt highest score first, sort gt ignore last
        gtind = np.argsort([g["_ignore"] for g in gt], kind="mergesort")
        gt = [gt[i] for i in gtind]
        dtind = np.argsort([-d["score"] for d in dt], kind="mergesort")
        dt = [dt[i] for i in dtind[0:max_det]]
        iscrowd = [int(o["iscrowd"]) for o in gt]
        # load computed ious
        ious = (
            self.ious[img_id, cat_id][:, gtind]
            if len(self.ious[img_id, cat_id]) > 0
            else self.ious[img_id, cat_id]
        )

        T = len(p.iou_thrs)
        G = len(gt)
        D = len(dt)
        gtm = np.zeros((T, G))
        dtm = np.zeros((T, D))
        gt_ious = np.zeros((T, G))
        dtIous = np.zeros((T, D))
        gt_ig = np.array([g["_ignore"] for g in gt])
        dt_ig = np.zeros((T, D))
        if not len(ious) == 0:
            for tind, t in enumerate(p.iou_thrs):
                for dind, d in enumerate(dt):
                    # information about best match so far (m=-1 -> unmatched)
                    iou = min([t, 1 - 1e-10])
                    m = -1
                    for gind, g in enumerate(gt):  # noqa B007
                        # if this gt already matched, and not a crowd, continue
                        if gtm[tind, gind] > 0 and not iscrowd[gind]:
                            continue
                        # if dt matched to reg gt, and on ignore gt, stop
                        if m > -1 and gt_ig[m] == 0 and gt_ig[gind] == 1:
                            break
                        # continue to next gt unless better match made
                        if ious[dind, gind] < iou:
                            continue
                        # if match successful and best so far, store appropriately # noqa
                        iou = ious[dind, gind]
                        m = gind
                    # if match made store id of match for both dt and gt
                    if m == -1:
                        continue
                    dt_ig[tind, dind] = gt_ig[m]
                    dtm[tind, dind] = gt[m]["id"]
                    gtm[tind, m] = d["id"]
                    dtIous[tind, dind] = iou
                    gt_ious[tind, m] = iou
        # set unmatched detections outside of area range to ignore
        a = np.array(
            [d["area"] < arng[0] or d["area"] > arng[1] for d in dt]
        ).reshape((1, len(dt)))
        dt_ig = np.logical_or(
            dt_ig, np.logical_and(dtm == 0, np.repeat(a, T, 0))
        )
        # store the max iou achiavable by every matched detection
        # and ground-truth
        dt_matches_max = []
        gt_matches_max = []
        r_dt_ious_max = [0.0 for d in dt] if check_scores else []
        r_gt_ious_max = [0.0 for g in gt] if check_scores else []

        gtNotIgnore = len([g for g in gt if g["_ignore"] == 0])
        # compute the optimal scores
        if check_scores and len(dt) != 0 and gtNotIgnore != 0:
            # there are both detections and ground truth annotations so an
            # optimal matching is required
            dt_m_max = np.zeros(D)
            dt_ious_max = np.zeros(D)
            gt_m_max = np.zeros(G)
            gt_ious_max = np.zeros(G)
            # give to every detection a score corresponding to the max
            # oks it could achieve with not-ignore ground-truth anns
            ious_mod = ious[:, :gtNotIgnore]

            dt_inds_max = [i for i in range(len(dt))]  # noqa C416
            gt_inds_max = np.argmax(ious_mod, axis=1).tolist()
            for i, (dtind, gtind) in enumerate(  # noqa B007
                zip(dt_inds_max, gt_inds_max)
            ):  # noqa B007
                dt_m_max[dtind] = gt[gtind]["id"]
                gt_m_max[gtind] = dt[dtind]["id"]
                dt_ious_max[dtind] = ious[dtind, gtind]
                gt_ious_max[gtind] = ious[dtind, gtind]

            dt_matches_max = [int(d) for d in dt_m_max]
            gt_matches_max = [int(g) for g in gt_m_max]
            r_dt_ious_max = dt_ious_max.tolist()
            r_gt_ious_max = gt_ious_max.tolist()

        # store results for given image and category
        return {
            "image_id": img_id,
            "category_id": cat_id,
            "aRng": arng,
            "maxDet": max_det,
            "dtIds": [d["id"] for d in dt],
            "gtIds": [g["id"] for g in gt],
            "dtMatches": dtm,
            "gtMatches": gtm,
            "dtScores": [d["score"] for d in dt],
            "gtIgnore": gt_ig,
            "dtIgnore": dt_ig,
            "dtIous": dtIous,
            "gtIous": gt_ious,
            "dtMatchesMax": dt_matches_max,
            "gtMatchesMax": gt_matches_max,
            "dtIousMax": r_dt_ious_max,
            "gtIousMax": r_gt_ious_max,
        }

    def accumulate(self, p=None):
        """Accumulate per image evaluation results and store the result in self.eval."""  # noqa
        logging.info("Accumulating evaluation results...")
        tic = time.time()
        if not self.eval_imgs:
            logging.info("Please run evaluate() first")
        # allows input customized parameters
        if p is None:
            p = self.params
        p.cat_ids = p.cat_ids if p.use_cats == 1 else [-1]
        T = len(p.iou_thrs)
        R = len(p.rec_thrs)
        K = len(p.cat_ids) if p.use_cats else 1
        A = len(p.area_rng)
        M = len(p.max_dets)
        precision = -np.ones(
            (T, R, K, A, M)
        )  # -1 for the precision of absent categories  # noqa
        recall = -np.ones((T, K, A, M))
        scores = -np.ones((T, R, K, A, M))

        # create dictionary for future indexing
        _pe = self._params_eval
        cat_ids = _pe.cat_ids if _pe.use_cats else [-1]
        setK = set(cat_ids)
        setA = set(map(tuple, _pe.area_rng))
        setM = set(_pe.max_dets)
        setI = set(_pe.img_ids)
        # get inds to evaluate
        k_list = [n for n, k in enumerate(p.cat_ids) if k in setK]
        m_list = [m for n, m in enumerate(p.max_dets) if m in setM]
        a_list = [
            n
            for n, a in enumerate(map(lambda x: tuple(x), p.area_rng))
            if a in setA
        ]
        i_list = [n for n, i in enumerate(p.img_ids) if i in setI]
        I0 = len(_pe.img_ids)
        A0 = len(_pe.area_rng)
        # retrieve E at each category, area range, and max number of detections
        for k, k0 in enumerate(k_list):
            Nk = k0 * A0 * I0
            for a, a0 in enumerate(a_list):
                Na = a0 * I0
                for m, max_det in enumerate(m_list):
                    E = [self.eval_imgs[Nk + Na + i] for i in i_list]
                    E = [e for e in E if e is not None]
                    if len(E) == 0:
                        continue
                    dtScores = np.concatenate(
                        [e["dtScores"][0:max_det] for e in E]
                    )

                    # different sorting method generates slightly different results.  # noqa
                    # mergesort is used to be consistent as Matlab implementation.  # noqa
                    inds = np.argsort(-dtScores, kind="mergesort")
                    dtScoresSorted = dtScores[inds]

                    dtm = np.concatenate(
                        [e["dtMatches"][:, 0:max_det] for e in E], axis=1
                    )[:, inds]
                    dt_ig = np.concatenate(
                        [e["dtIgnore"][:, 0:max_det] for e in E], axis=1
                    )[:, inds]
                    gt_ig = np.concatenate([e["gtIgnore"] for e in E])
                    npig = np.count_nonzero(gt_ig == 0)
                    if npig == 0:
                        continue
                    tps = np.logical_and(dtm, np.logical_not(dt_ig))  # noqa
                    fps = np.logical_and(
                        np.logical_not(dtm), np.logical_not(dt_ig)
                    )

                    tp_sum = np.cumsum(tps, axis=1).astype(dtype=np.float64)
                    fp_sum = np.cumsum(fps, axis=1).astype(dtype=np.float64)
                    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                        tp = np.array(tp)
                        fp = np.array(fp)
                        nd = len(tp)
                        rc = tp / npig
                        pr = tp / (fp + tp + np.spacing(1))
                        q = np.zeros((R,))
                        ss = np.zeros((R,))

                        if nd:
                            recall[t, k, a, m] = rc[-1]
                        else:
                            recall[t, k, a, m] = 0

                        # numpy is slow without cython optimization for accessing elements  # noqa
                        # use python array gets significant speed improvement
                        pr = pr.tolist()
                        q = q.tolist()

                        for i in range(nd - 1, 0, -1):
                            if pr[i] > pr[i - 1]:
                                pr[i - 1] = pr[i]

                        inds = np.searchsorted(rc, p.rec_thrs, side="left")
                        try:
                            for ri, pi in enumerate(inds):
                                q[ri] = pr[pi]
                                ss[ri] = dtScoresSorted[pi]
                        except Exception:
                            pass
                        precision[t, :, k, a, m] = np.array(q)
                        scores[t, :, k, a, m] = np.array(ss)
        self.eval = {
            "params": p,
            "counts": [T, R, K, A, M],
            "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "precision": precision,
            "recall": recall,
            "scores": scores,
        }
        toc = time.time()
        logging.info("DONE (t={:0.2f}s).".format(toc - tic))

    def summarize(self):
        """Compute and display summary metrics for evaluation results."""

        def _summarize(ap=1, iou_thr=None, area_rng="all", max_dets=100):
            p = self.params
            iStr = " {:<18} {} @[ IoU={:<9} | area={:>6s} | max_dets={:>3d} ] = {:0.3f}"  # noqa
            titleStr = "Average Precision" if ap == 1 else "Average Recall"
            typeStr = "(AP)" if ap == 1 else "(AR)"
            iouStr = (
                "{:0.2f}:{:0.2f}".format(p.iou_thrs[0], p.iou_thrs[-1])
                if iou_thr is None
                else "{:0.2f}".format(iou_thr)
            )

            aind = [
                i for i, arng in enumerate(p.area_rng_lbl) if arng == area_rng
            ]
            mind = [i for i, mDet in enumerate(p.max_dets) if mDet == max_dets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval["precision"]
                # IoU
                if iou_thr is not None:
                    t = np.where(iou_thr == p.iou_thrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]  # noqa
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval["recall"]
                if iou_thr is not None:
                    t = np.where(iou_thr == p.iou_thrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            logging.info(
                iStr.format(
                    titleStr, typeStr, iouStr, area_rng, max_dets, mean_s
                )
            )  # noqa
            return mean_s

        def _summarize_dets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(
                1, iou_thr=0.5, max_dets=self.params.max_dets[2]
            )
            stats[2] = _summarize(
                1, iou_thr=0.75, max_dets=self.params.max_dets[2]
            )
            stats[3] = _summarize(
                1, area_rng="small", max_dets=self.params.max_dets[2]
            )
            stats[4] = _summarize(
                1, area_rng="medium", max_dets=self.params.max_dets[2]
            )
            stats[5] = _summarize(
                1, area_rng="large", max_dets=self.params.max_dets[2]
            )
            stats[6] = _summarize(0, max_dets=self.params.max_dets[0])
            stats[7] = _summarize(0, max_dets=self.params.max_dets[1])
            stats[8] = _summarize(0, max_dets=self.params.max_dets[2])
            stats[9] = _summarize(
                0, area_rng="small", max_dets=self.params.max_dets[2]
            )
            stats[10] = _summarize(
                0, area_rng="medium", max_dets=self.params.max_dets[2]
            )
            stats[11] = _summarize(
                0, area_rng="large", max_dets=self.params.max_dets[2]
            )
            return stats

        def _summarize_ldmk():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, max_dets=20)
            stats[1] = _summarize(1, max_dets=20, iou_thr=0.5)
            stats[2] = _summarize(1, max_dets=20, iou_thr=0.75)
            stats[3] = _summarize(1, max_dets=20, area_rng="medium")
            stats[4] = _summarize(1, max_dets=20, area_rng="large")
            stats[5] = _summarize(0, max_dets=20)
            stats[6] = _summarize(0, max_dets=20, iou_thr=0.5)
            stats[7] = _summarize(0, max_dets=20, iou_thr=0.75)
            stats[8] = _summarize(0, max_dets=20, area_rng="medium")
            stats[9] = _summarize(0, max_dets=20, area_rng="large")
            return stats

        if not self.eval:
            raise Exception("Please run accumulate() first")
        iou_type = self.params.iou_type
        if iou_type == "segm" or iou_type == "bbox":
            summarize = _summarize_dets
        elif iou_type == "keypoints":
            summarize = _summarize_ldmk
        self.stats = summarize()

    def __str__(self):
        self.summarize()


class Params:
    """Params for coco evaluation api.

    Args:
        iou_type: Iou type. Defaults to "segm".
        ldmk_type: ldmk type. Defaults to "hobot_kps17".
    """

    def __init__(
        self,
        iou_type="segm",
        ldmk_type="hobot_kps17",
    ):
        if iou_type == "segm" or iou_type == "bbox":
            self.set_det_params()
        elif iou_type == "keypoints":
            assert ldmk_type in ["hobot_kps17", "hobot_kps19", "cockpit_kps15"]
            self.set_ldmk_params(ldmk_type=ldmk_type)
        else:
            raise Exception("iou_type not supported")
        self.iou_type = iou_type
        # use_segm is deprecated
        self.use_segm = None

    def set_det_params(self):
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value  # noqa
        self.iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = [1, 10, 100]
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],
            [0 ** 2, 32 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.area_rng_lbl = ["all", "small", "medium", "large"]
        self.use_cats = 1

    def set_ldmk_params(self, ldmk_type):
        self.img_ids = []
        self.cat_ids = []
        # np.arange causes trouble.  the data point on arange is slightly larger than the true value  # noqa
        self.iou_thrs = np.linspace(
            0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
        )
        self.rec_thrs = np.linspace(
            0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True
        )
        self.max_dets = [20]
        self.area_rng = [
            [0 ** 2, 1e5 ** 2],
            [32 ** 2, 96 ** 2],
            [96 ** 2, 1e5 ** 2],
        ]
        self.area_rng_lbl = ["all", "medium", "large"]
        self.use_cats = 1
        if ldmk_type == "hobot_kps17":
            self.kpt_oks_sigmas = (
                np.array(
                    [
                        0.26,
                        0.25,
                        0.25,
                        0.35,
                        0.35,
                        0.79,
                        0.79,
                        0.72,
                        0.72,
                        0.62,
                        0.62,
                        1.07,
                        1.07,
                        0.87,
                        0.87,
                        0.89,
                        0.89,
                    ]
                )
                / 10.0
            )
        elif ldmk_type == "hobot_kps19":
            # https://github.com/jin-s13/COCO-WholeBody/blob/master/evaluation/myeval_lefthand.py#L171
            self.kpt_oks_sigmas = (
                np.array(
                    [
                        0.26,
                        0.25,
                        0.25,
                        0.35,
                        0.35,
                        0.79,
                        0.79,
                        0.72,
                        0.72,
                        0.62,
                        0.62,
                        1.07,
                        1.07,
                        0.87,
                        0.87,
                        0.89,
                        0.89,
                        0.24,
                        0.24,
                    ]
                )
                / 10.0
            )
        elif ldmk_type == "cockpit_kps15":
            self.kpt_oks_sigmas = (
                np.array(
                    [
                        1.07,
                        0.87,
                        0.89,
                        1.07,
                        0.87,
                        0.89,
                        0.79,
                        0.72,
                        0.62,
                        0.24,
                        0.79,
                        0.72,
                        0.62,
                        0.24,
                        0.26,
                    ]
                )
                / 10.0
            )


class COCOEval(object):
    """COCO eval.

    Args:
        anno_path: Annotation path.
        choose_image_ids: Image id you want to select. Defaults to None.
    """

    @require_packages("pycocotools")
    def __init__(self, anno_path, choose_image_ids=None):
        self.coco = COCO(anno_path)
        self.name = anno_path[:-5].split("_")[-1]

        self.image_ids = self.coco.getImgIds()
        if choose_image_ids is not None:
            self.image_ids = [self.image_ids[i] for i in choose_image_ids]

        self.cat_ids = self.coco.getCatIds()
        self.cat_names = [
            cat["name"] for cat in self.coco.loadCats(self.cat_ids)
        ]
        self.stuffStartId = np.min(self.cat_ids)
        self.stuffEndId = np.max(self.cat_ids)
        self._gt_anno_path = anno_path

    def _eval(self, coco_eval, task_name, eval_cls=-1):
        coco_eval.params.img_ids = self.image_ids

        def catch_print():
            import io

            sys.stdout = io.StringIO()
            coco_eval.summarize()
            coco_summary = sys.stdout.getvalue()
            logging.info("\n{}".format(coco_summary))

        if isinstance(eval_cls, (list, tuple)):
            for c in eval_cls:
                assert c > 0
                coco_eval.params.cat_ids = [self.cat_ids[c - 1]]
                coco_eval.evaluate()
                coco_eval.accumulate()
                logging.info(
                    "%s %s result:" % (self.cat_names[c - 1], task_name)
                )
                catch_print()
        else:
            if eval_cls == -1:
                coco_eval.evaluate()
                coco_eval.accumulate()
                logging.info("%s result:" % task_name)
                catch_print()
            elif eval_cls == -2:
                for i, cat_id in enumerate(self.cat_ids):
                    coco_eval.params.cat_ids = [cat_id]
                    coco_eval.evaluate()
                    coco_eval.accumulate()
                    logging.info(
                        "%s %s result:" % (self.cat_names[i], task_name)
                    )
                    catch_print()
            elif eval_cls > 0:
                coco_eval.params.cat_ids = [self.cat_ids[eval_cls - 1]]
                coco_eval.evaluate()
                coco_eval.accumulate()
                logging.info(
                    "%s %s result:" % (self.cat_names[eval_cls - 1], task_name)
                )
                catch_print()
            else:
                assert False  # noqa B011

    def write_ldmk_results(self, pred_results, res_file):
        assert len(self.image_ids) == len(pred_results)
        results = []
        for image_id, pred_result in zip(self.image_ids, pred_results):
            if not pred_result:
                continue
            pred_ldmk = pred_result["pred_ldmk"]
            pred_ldmk_scores = pred_result["pred_ldmk_scores"]
            pred_ldmk_classes = np.ones_like(pred_ldmk_scores)
            num_boxes = len(pred_ldmk)
            if num_boxes > 0:
                for i in range(num_boxes):
                    res = dict()  # noqa C408
                    res["image_id"] = image_id
                    if pred_ldmk_classes[i] in self.cat_ids:
                        idx = self.cat_ids.index(pred_ldmk_classes[i])
                        res["category_id"] = self.cat_ids[idx]
                    else:
                        continue
                    res["keypoints"] = list(pred_ldmk[i])
                    res["score"] = float(pred_ldmk_scores[i])
                    assert res["category_id"] == 1
                    results.append(res)
        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)
        return results

    def write_det_results(self, pred_results, res_file):
        assert len(self.image_ids) == len(pred_results)
        results = []
        for image_id, pred_result in zip(self.image_ids, pred_results):
            if not pred_result:
                continue
            pred_boxes = pred_result["pred_boxes"]
            pred_scores = pred_result["pred_scores"]
            pred_classes = np.ones_like(pred_scores)
            num_boxes = len(pred_boxes)
            if num_boxes > 0:
                for i in range(num_boxes):
                    box = pred_boxes[i]
                    w = box[2] - box[0] + 1
                    h = box[3] - box[1] + 1
                    res = dict()  # noqa C408
                    res["bbox"] = [box[0], box[1], w, h]
                    res["image_id"] = image_id
                    if pred_classes[i] in self.cat_ids:
                        idx = self.cat_ids.index(pred_classes[i])
                        res["category_id"] = self.cat_ids[idx]
                    else:
                        continue
                    res["score"] = pred_scores[i]
                    results.append(res)
        with open(res_file, "w") as f:
            json.dump(results, f, sort_keys=True, indent=4)
        return results

    def eval_ldmk(self, pred_results, res_file=None, eval_cls=1):
        if res_file is None:
            res_file = "person_keypoints_%s_pid%d_results.json" % (
                self.name,
                os.getpid(),
            )
        results = self.write_ldmk_results(pred_results, res_file)
        if len(results) == 0:
            logging.info("no detection, skip keypoints evaluation")
            os.system("rm %s" % res_file)
            return
        if "test" not in self.name:
            coco_res = self.coco.loadRes(res_file)
            coco_eval = HobotCOCOeval(self.coco, coco_res, "keypoints")
            self._eval(coco_eval, "keypoints", eval_cls=eval_cls)
            os.system("rm %s" % res_file)
        return coco_eval

    def eval_det(self, pred_results, res_file=None, eval_cls=-1):
        if res_file is None:
            res_file = "detections_%s_pid%d_results.json" % (
                self.name,
                os.getpid(),
            )
        results = self.write_det_results(pred_results, res_file)
        if len(results) == 0:
            logging.info("no detection, skip detection evaluation")
            os.system("rm %s" % res_file)
            return
        if "test" not in self.name:
            coco_res = self.coco.loadRes(res_file)
            coco_eval = HobotCOCOeval(self.coco, coco_res, "bbox")
            self._eval(coco_eval, "detection", eval_cls=eval_cls)
            os.system("rm %s" % res_file)
            coco_eval.accumulate()
            coco_eval.evaluate()


@OBJECT_REGISTRY.register
class COCOLdmkMetric(EvalMetric):
    """Evaluation ldmk in COCO protocol.

    This class reads data in a distributed manner when using multiple gpus.
    And call the COCO evaluation tool in rank 0.

    Args:
        ann_file: Validation data annotation json file path.
        save_prefix: Path to save result.Defaults to "./tmp_results".
        use_time: Whether to use time for name.Defaults to True.
        cleanup: Whether to clean up the saved results. Defaults to True.
    """

    def __init__(
        self,
        ann_file: str,
        save_prefix: str = "./tmp_results",
        use_time: bool = True,
        cleanup: bool = True,
    ):
        name = ["ap", "ap50", "ap75", "ar", "ar50", "ar75"]
        super(COCOLdmkMetric, self).__init__(name)
        self.ann_file = ann_file
        self.save_prefix = save_prefix
        self.use_time = use_time
        self.cleanup = cleanup

        try:
            os.makedirs(osp.expanduser(self.save_prefix))
        except Exception:
            pass

        if use_time:
            t = datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S")
        else:
            t = ""
        self._filename = osp.abspath(
            osp.join(osp.expanduser(self.save_prefix), t + ".json")
        )
        try:
            f = open(self._filename, "w")
        except IOError as e:
            raise RuntimeError(
                "Unable to open json file to dump. What(): {}".format(str(e))
            )
        else:
            f.close()

        self.reset()

    def _init_states(self):
        self.add_state(
            "_results",
            default=[],
        )

    def __del__(self):
        if self.cleanup:
            try:
                os.remove(self._filename)
            except IOError as err:
                logger.error(str(err))

    def _remove_redundance_result(self, results):
        img_name_list = []
        new_result = []
        for ret in results:
            image_name = ret["image_name"]
            if image_name not in img_name_list:
                img_name_list.append(image_name)
                new_result.append(ret)
            else:
                logging.warn(
                    f"find redundance img {image_name} result, remove it now"
                )
        return new_result

    def _update_coco_results(self, results):
        results = self._remove_redundance_result(results)
        coco_eval = COCOEval(self.ann_file)

        img_names = []
        for i in range(len(coco_eval.coco.imgs)):
            img_names.append(coco_eval.coco.imgs[i + 1]["file_name"])
        pred_results_map = {}
        for pred_res in results:
            image_name = pred_res["image_name"].split("/")[-1]
            pred_results_map[image_name] = pred_res
        pred_results = []
        for image_name in img_names:
            if os.path.basename(image_name) in pred_results_map:
                pred_results.append(
                    pred_results_map[os.path.basename(image_name)]
                )
            else:
                pred_results.append(dict())  # noqa C408

        _coco_eval = coco_eval.eval_ldmk(
            pred_results, self._filename, eval_cls=1
        )
        coco_eval.eval_det(pred_results, eval_cls=-2)
        _coco_eval.evaluate()
        _coco_eval.accumulate()

        return _coco_eval

    def get_value(self, coco_eval):
        iou_50_index = np.where(coco_eval.params.iou_thrs == 0.5)[0]
        iou_75_index = np.where(coco_eval.params.iou_thrs == 0.75)[0]
        area_index = coco_eval.params.area_rng_lbl.index("all")
        maxdet_index = coco_eval.params.max_dets.index(20)
        ap = coco_eval.eval["precision"][
            :, :, :, area_index, maxdet_index
        ].mean()
        ap50 = coco_eval.eval["precision"][
            iou_50_index, :, :, area_index, maxdet_index
        ].mean()
        ap75 = coco_eval.eval["precision"][
            iou_75_index, :, :, area_index, maxdet_index
        ].mean()

        ar = coco_eval.eval["recall"][:, :, area_index, maxdet_index].mean()
        ar50 = coco_eval.eval["recall"][
            iou_50_index, :, area_index, maxdet_index
        ].mean()
        ar75 = coco_eval.eval["recall"][
            iou_75_index, :, area_index, maxdet_index
        ].mean()

        values = [ap, ap50, ap75, ar, ar50, ar75]
        return values

    def compute(self):
        """Get evaluation metrics."""
        results = []
        values = [0, 0, 0, 0, 0, 0]
        # if distributed is used, gather data from all process.
        if dist.is_initialized():
            dist.barrier()
            world_size = dist.get_world_size()
            gather_data = [None for _ in range(world_size)]
            # gather results from all process.
            dist.all_gather_object(gather_data, self._results)
            for data in gather_data:
                for item in data:
                    results.append(item)

            # do val only on rank 0
            if dist.get_rank() == 0:
                coco_eval = self._update_coco_results(results)
                values = self.get_value(coco_eval)
            self.reset()
            return values

        else:
            results = self._results
            coco_eval = self._update_coco_results(results)
            values = self.get_value(coco_eval)
            self.reset()

            return values

    def get_ldmk_score(self, ldmk, box_score):
        ldmk_each_scores = ldmk[:, 2::3].copy()
        ldmk_each_scores[ldmk_each_scores < 0.1] = 0
        ldmk_scores_sum = ldmk_each_scores.sum(axis=1)
        ldmk_each_scores[ldmk_each_scores > 0] = 1
        ldmk_scores_count = ldmk_each_scores.sum(axis=1)
        ldmk_scores_count[ldmk_scores_count == 0] = 1
        ldmk_score = ldmk_scores_sum / ldmk_scores_count
        ldmk_score = (ldmk_score + box_score) / 2.0
        return ldmk_score

    def update(self, batch, preds):
        batch_data = batch

        image_name = batch_data["image_name"]
        batch_size = len(image_name)

        pred_ldmk = preds["pred_ldmk"]
        pred_boxes_scores = preds["pred_scores"]
        pred_boxes = preds["pred_boxes"]

        for idx in range(batch_size):
            if "scale_factor" in batch_data:
                pred_ldmk[idx][:, 0::3] = (
                    pred_ldmk[idx][:, 0::3]
                    / batch_data["scale_factor"][idx][0].item()  # noqa
                )
                pred_ldmk[idx][:, 1::3] = (
                    pred_ldmk[idx][:, 1::3]
                    / batch_data["scale_factor"][idx][1].item()  # noqa
                )
                pred_boxes[idx][:, 0] = (
                    pred_boxes[idx][:, 0]
                    / batch_data["scale_factor"][idx][0].item()  # noqa
                )
                pred_boxes[idx][:, 2] = (
                    pred_boxes[idx][:, 2]
                    / batch_data["scale_factor"][idx][0].item()  # noqa
                )
                pred_boxes[idx][:, 1] = (
                    pred_boxes[idx][:, 1]
                    / batch_data["scale_factor"][idx][1].item()  # noqa
                )
                pred_boxes[idx][:, 3] = (
                    pred_boxes[idx][:, 3]
                    / batch_data["scale_factor"][idx][1].item()  # noqa
                )
            result = {}
            result["image_name"] = image_name[idx]
            result["pred_ldmk"] = (
                pred_ldmk[idx].detach().cpu().numpy().tolist()
            )
            result["pred_ldmk_scores"] = self.get_ldmk_score(
                pred_ldmk[idx].detach().cpu().numpy(),
                pred_boxes_scores[idx].detach().cpu().numpy(),
            ).tolist()
            result["pred_boxes"] = (
                pred_boxes[idx].detach().cpu().numpy().tolist()
            )
            result["pred_scores"] = (
                pred_boxes_scores[idx].detach().cpu().numpy().tolist()
            )
            self._results.append(result)
