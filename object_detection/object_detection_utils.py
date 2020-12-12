import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common


class CustomCocoEval(COCOeval):
    def computeAreaRng(self, large_thresh=.68, max_fact=1.1):
        """Returns range for ['all', 'small', 'medium', 'large']
        large min boundary is taken so that large_thresh % of items are smaller
        max fact used to increase max object boundary in comparison to gt"""
        areas = np.array([a['area'] for a in self.cocoGt.anns.values()])
        lmin, lmax = np.quantile(areas, large_thresh).astype(np.int), int(np.max(areas) * max_fact)
        return [[0, lmax], [0, lmin//2], [lmin//2, lmin], [lmin, lmax]]

    def computeMaxDets(self, facts=(1.25, 1.5)):
        """The min dets will be the max amount of objs in any GT img and the value obtained after multiplying facts"""
        dets = np.array([len(img_anns) for img_anns in self.cocoGt.imgToAnns.values()])
        max_dets = np.max(dets)
        return ("x1", *tuple(f"x{f}" for f in facts)), [max_dets, *tuple(int(f*max_dets) for f in facts)]

    def __init__(self, *args, all_cats="all", **kwargs):
        super().__init__(*args, **kwargs)
        self.cats = [all_cats] + [self.cocoGt.cats[cid]['name'] for cid in self.params.catIds]
        self.ious_summary = [.15, .25, .5, .75]
        self.params.iouThrs = np.linspace(.15, 0.95, np.int(np.round((0.95 - .15) / .05)) + 1, endpoint=True)
        self.params.areaRng = self.computeAreaRng()
        self.params.maxDetsLbl, self.params.maxDets = self.computeMaxDets()

    def getPrecisionRecall(self):
        """Returns eval labels and results, each of shape (2, nbIoUs, nbCats, nbAreasRngs, nbMaxDets)"""
        p = self.params
        labels = tuple(map(np.array, (("Precision", "Recall"), p.iouThrs, self.cats, p.areaRngLbl, p.maxDetsLbl)))
        # average precision over all recall thresholds
        pre, rec = self.eval['precision'].mean(axis=1), self.eval['recall']
        pre_w_all_cats = np.concatenate((pre.mean(axis=1, keepdims=True), pre), axis=1)
        rec_w_all_cats = np.concatenate((rec.mean(axis=1, keepdims=True), rec), axis=1)
        return np.array(labels), np.stack((pre_w_all_cats, rec_w_all_cats))

    def eval_acc_and_summarize(self, verbose=True):
        self.evaluate()
        self.accumulate()
        self.summarize(verbose)

    def summarize(self, verbose):
        if not self.eval:
            raise Exception('Please run accumulate() first')

        p = self.params

        def _iou_res(iouThr, s):
            return s if iouThr is None else s[iouThr == p.iouThrs]

        def _summarize(iouThr=None, areaRng=p.areaRngLbl[0], maxDets=p.maxDets[0], cat=self.cats[0]):
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            # first cats item is all_cats with id 0 (this id does not exist in self.eval)
            cind = [i for i, c in enumerate(p.catIds) if self.cats[c] == cat or cat == self.cats[0]]

            # dimension of precision: [TxRxKxAxM]
            sap = _iou_res(iouThr, self.eval['precision'])[:, :, cind, aind, mind]
            mean_sap = -1 if not len(sap[sap > -1]) else np.mean(sap[sap > -1])

            # dimension of recall: [TxKxAxM]
            sar = _iou_res(iouThr, self.eval['recall'])[:, cind, aind, mind]
            mean_sar = -1 if not len(sar[sar > -1]) else np.mean(sar[sar > -1])

            return mean_sap, mean_sar

        self.stats = np.zeros((2, len(self.cats), len(p.areaRng), len(p.maxDets), len(self.ious_summary) + 1), np.float)
        res_str = ""
        for ci, cat in enumerate(self.cats):
            res_str += f"CATEGORY: {cat}\n"
            for ai, (area, _) in enumerate(zip(p.areaRngLbl, p.areaRng)):
                res_str += f"\tOBJECT SIZE: {area}\n"
                for di, det in enumerate(p.maxDets):
                    res_str += f"\t\tMAX DET COUNT: {det}\n"
                    for ii, iou in enumerate([*self.ious_summary, None]):
                        iouStr = f'{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}' if iou is None else f'{iou:0.2f}'
                        ap, ar = _summarize(iou, area, det, cat)
                        res_str += f"\t\t\tIoU={iouStr:<10}: AP={ap:6.3f}, AR={ar:6.3f}\n"
                        self.stats[:, ci, ai, di, ii] = ap, ar
        if verbose: print(res_str)

    @staticmethod
    def plot_coco_eval(eval_stats_labels, eval_stats_with_err, figsize, save_path=None, show_val=False):
        # shape (2, nbIoUs, nbCats, nbAreasRngs, nbMaxDets)
        pre_rec, pre_rec_err = eval_stats_with_err
        pre_rec_labels, ious, cats, areaRngLbl, maxDetsLbl = eval_stats_labels

        base = slice(None), slice(None)
        # idx_slices are ([:, :, :, 0, -1], [:, :, 0, :, -1], [:, :, 0, 0, :])
        idx_slices = ((*base, slice(None), 0, -1), (*base, 0, slice(None), -1), (*base, 0, 0, slice(None)))
        idx_slice_titles = []
        for idxs in idx_slices:
            for si, lab, param in zip(idxs[2:], eval_stats_labels[2:], ("cats", "areaRng", "maxDets")):
                if si != slice(None): idx_slice_titles.append(" ".join(f"{lab[si]} {param}"))

        fig, axs = plt.subplots(3, 2, figsize)
        for laxs in axs:
            for labels, idxs, idxs_title in zip(eval_stats_labels[2:], idx_slices, idx_slice_titles):
                xiou = ious[None].repeat(labels.size, axis=0)
                for ax, vname, values, err in zip(laxs, pre_rec_labels, pre_rec[idxs], pre_rec_err[idxs]):
                    common.plot_lines_with_err(ax, xiou, values, err, labels, show_val)
                    ax.set_title(f'{vname} with {idxs_title}')
                    ax.set_xlabel("IoU thresholds")
                    ax.set_ylabel("Percentage")
        fig.tight_layout(pad=.2)
        if save_path is not None:
            plt.savefig(save_path, dpi=400)

