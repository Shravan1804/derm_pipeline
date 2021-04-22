import os
import sys

import numpy as np
import matplotlib.pyplot as plt

import icevision.all as ia
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from segmentation import mask_utils
from object_detection import coco_format


def get_obj_det_ds_paths(root, anno_file):
    anno_path = os.path.join(root, 'annotations', anno_file)
    img_dir = os.path.join(root, 'images', os.path.splitext(anno_file)[0])
    return anno_path, img_dir


class COCOMaskParserEncrypted(ia.parsers.COCOMaskParser):
    def image_width_height(self, o):
        im_dict = self._imageid2info[o['image_id']]
        return im_dict["width"], im_dict["height"]


class SimpleCocoEvalMetric(COCOeval):
    def __init__(self, coco_eval, cat_id=None, iou=.15, pre_rec=True):
        super().__init__(coco_eval.cocoGt, coco_eval.cocoDt, coco_eval.params.iouType)
        self.cat_id = cat_id
        self.iou = iou
        self.pre_rec = pre_rec
        self.params.iouThrs = np.array([iou])
        self.params.areaRng = [[0, int(1e10)]]
        self.params.maxDets = [int(1e3)]

    def summarize(self):
        p = self.params
        def _iou_res(iouThr, s): return s if iouThr is None else s[iouThr == p.iouThrs]
        def _summarize():
            aind, mind = [0], [0]
            cind = [i for i, c in enumerate(p.catIds) if self.cat_id == c or self.cat_id is None]
            # dimension of precision: [TxRxKxAxM]
            sap = _iou_res(self.iou, self.eval['precision'])[:, :, cind, aind, mind]
            mean_sap = -1 if not len(sap[sap > -1]) else np.mean(sap[sap > -1])
            # dimension of recall: [TxKxAxM]
            sar = _iou_res(self.iou, self.eval['recall'])[:, cind, aind, mind]
            mean_sar = -1 if not len(sar[sar > -1]) else np.mean(sar[sar > -1])
            return mean_sap, mean_sar
        ap, ar = _summarize()
        self.logs = {'val': ap if self.pre_rec else ar}
        return self.logs


class CustomCOCOMetric(ia.COCOMetric):
    def __init__(self, cat_id, iou, metric="precision", metric_type=ia.COCOMetricType.bbox):
        super().__init__(metric_type)
        self.cat_id = cat_id
        self.iou = iou
        self.pre_rec = metric == "precision"

    def finalize(self):
        with ia.CaptureStdout():
            coco_eval = ia.create_coco_eval(records=self._records, preds=self._preds,
                                            metric_type=self.metric_type.value, show_pbar=self.show_pbar)
            coco_eval = SimpleCocoEvalMetric(coco_eval, self.cat_id, self.iou, self.pre_rec)
            coco_eval.evaluate()
            coco_eval.accumulate()
        with ia.CaptureStdout(propagate_stdout=self.print_summary):
            coco_eval.summarize()
        logs = coco_eval.logs
        self._reset()
        return logs


def segm_dataset_to_coco_format(segm_masks, cats, scores=False, bg=0, ret_json=False):
    segm_masks = segm_masks if type(segm_masks) is np.ndarray else segm_masks.numpy()
    dataset = coco_format.get_default_dataset()
    cats = [(c, i) for i, c in enumerate(cats) if i != bg]
    dataset['categories'] = coco_format.get_categories(*zip(*cats))
    ann_id = 1  # MUST start at 1 since pycocotools.cocoeval uses detId to track matches and checks with > 0
    for img_id, non_binary_mask in enumerate(segm_masks):
        img_id += 1  # to be on the safe side (same idea as ann_id)
        dataset['images'].append(coco_format.get_img_record(img_id, f'{img_id}.jpg', non_binary_mask.shape))
        obj_cats_with_masks = mask_utils.separate_objs_in_mask(non_binary_mask, bg=bg)
        if obj_cats_with_masks is None: continue
        else: obj_cats, obj_cats_masks = obj_cats_with_masks
        ann_id, img_annos = coco_format.get_annos_from_objs_mask(img_id, ann_id, obj_cats, obj_cats_masks, scores)
        dataset['annotations'].extend(img_annos)
    if ret_json:
        return dataset
    else:
        coco_ds = COCO()
        coco_ds.dataset = dataset
        coco_ds.createIndex()
        return coco_ds


class CustomCocoEval(COCOeval):
    def computeAreaRng(self, large_thresh=.68, max_fact=1.1):
        """Returns range for ['all', 'small', 'medium', 'large']
        large min boundary is taken so that large_thresh % of items are smaller
        max fact used to increase max object boundary in comparison to gt"""
        areas = np.array([a['area'] for a in self.cocoGt.anns.values()])
        lmin, lmax = np.quantile(areas, large_thresh).astype(np.int), int(np.max(areas) * max_fact)
        return [[0, lmax], [0, lmin//2], [lmin//2, lmin], [lmin, lmax]]

    def computeMaxDets(self, facts=(1, 1.5, 2)):
        """The min dets will be the max amount of objs in any GT img multiplied by provided facts"""
        dets = np.array([len(img_anns) for img_anns in self.cocoGt.imgToAnns.values()])
        max_dets = np.max(dets)
        return tuple(f"x{f}" for f in facts), tuple(int(f*max_dets) for f in facts)

    def __init__(self, *args, cats=None, all_cats="all", **kwargs):
        super().__init__(*args, **kwargs)
        self.cats = [all_cats] + [self.cocoGt.cats[cid]['name'] for cid in self.params.catIds] if cats is None else cats
        self.ious_summary = [.15, .25, .5, .75]
        self.params.iouThrs = np.linspace(.15, 0.95, np.int(np.round((0.95 - .15) / .05)) + 1, endpoint=True)
        self.params.areaRng = self.computeAreaRng()
        self.params.maxDetsLbl, self.params.maxDets = self.computeMaxDets()

    def get_precision_recall_with_labels(self):
        """Returns eval labels and results, each of shape (2, nbIoUs, nbCats, nbAreasRngs, nbMaxDets)"""
        p = self.params
        labels = tuple(map(np.array, (("Precision", "Recall"), p.iouThrs, self.cats, p.areaRngLbl, p.maxDetsLbl)))
        labels = np.array(labels, dtype=np.object)
        areaRng, maxDets = map(np.array, (p.areaRng, p.maxDets))
        # average precision over all recall thresholds
        pre, rec = self.eval['precision'].mean(axis=1), self.eval['recall']
        pre_w_all_cats = np.concatenate((pre.mean(axis=1, keepdims=True), pre), axis=1)
        rec_w_all_cats = np.concatenate((rec.mean(axis=1, keepdims=True), rec), axis=1)
        return labels, areaRng, maxDets, np.stack((pre_w_all_cats, rec_w_all_cats))

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
            cind = [i for i, c in enumerate(p.catIds) if cat == self.cats[c] or cat == self.cats[0]]

            # dimension of precision: [TxRxKxAxM]
            sap = _iou_res(iouThr, self.eval['precision'])[:, :, cind, aind, mind]
            mean_sap = -1 if not len(sap[sap > -1]) else np.mean(sap[sap > -1])

            # dimension of recall: [TxKxAxM]
            sar = _iou_res(iouThr, self.eval['recall'])[:, cind, aind, mind]
            mean_sar = -1 if not len(sar[sar > -1]) else np.mean(sar[sar > -1])

            return mean_sap, mean_sar

        self.stats = np.zeros((2, len(self.cats), len(p.areaRng), len(p.maxDets), len(self.ious_summary) + 1), np.float)
        self.logs = {}
        res_str = ""
        for ci, cat in enumerate(self.cats):
            res_str += f"CATEGORY: {cat}\n"
            log_key = cat
            for ai, (area, _) in enumerate(zip(p.areaRngLbl, p.areaRng)):
                res_str += f"\tOBJECT SIZE: {area}\n"
                log_key += '_' + area
                for di, (detLbl, det) in enumerate(zip(p.maxDetsLbl, p.maxDets)):
                    res_str += f"\t\tMAX DET COUNT {detLbl}: {det}\n"
                    log_key += '_' + detLbl + '_' + str(det)
                    for ii, iou in enumerate([*self.ious_summary, None]):
                        iouStr = f'{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}' if iou is None else f'{iou:0.2f}'
                        ap, ar = _summarize(iou, area, det, cat)
                        res_str += f"\t\t\tIoU={iouStr:<10}: AP={ap:6.3f}, AR={ar:6.3f}\n"
                        log_key += '_' + iouStr
                        self.stats[:, ci, ai, di, ii] = ap, ar
                        self.logs[log_key + '__AP'] = ap
                        self.logs[log_key + '__AR'] = ar
        if verbose: print(res_str)

    @staticmethod
    def plot_coco_eval(eval_stats_labels, eval_stats_with_err, figsize, save_path=None, show_val=False):
        # shape (2, nbIoUs, nbCats, nbAreasRngs, nbMaxDets)
        pre_rec, pre_rec_err = eval_stats_with_err
        pre_rec_labels, ious, cats, areaRngLbl, maxDetsLbl = eval_stats_labels

        all_labels = cats, *(areaRngLbl, maxDetsLbl) * len(cats)
        A = slice(None)
        # idx_slices are ([:, :, :, 0, -1], [:, :, 0, :, -1], [:, :, 0, 0, :])
        # idx_slices = (A, A, A, 0, -1), (A, A, 0, A, -1), (A, A, 0, 0, A)
        idx_slices = (i for ii in (((A, A, ci, A, -1), (A, A, ci, 0, A)) for ci in range(len(cats))) for i in ii)
        idx_slices = (A, A, A, 0, -1), *idx_slices
        # idx_slices_codes = "cats", "areaRng", "maxDets"
        idx_slices_codes = (i for ii in ((f'areaRng_{c}', f'maxDets_{c}') for c in cats) for i in ii)
        idx_slices_codes = "cats", *idx_slices_codes
        # idx_slice_titles = "all sizes and maxDets", "all cats and maxDets", "all cats and sizes"
        idx_slice_titles = (i for ii in ((f'{c} cat & maxDets', f'{c} cat & all sizes') for c in cats) for i in ii)
        idx_slice_titles = "all sizes & maxDets", *idx_slice_titles

        for labels, idxs, code, title in zip(all_labels, idx_slices, idx_slices_codes, idx_slice_titles):
            fig, axs = plt.subplots(1, 2, figsize=figsize)
            xiou = ious[None].repeat(labels.size, axis=0)
            for ax, vname, values, err in zip(axs, pre_rec_labels, pre_rec[idxs], pre_rec_err[idxs]):
                values, err = np.moveaxis(values, 0, -1), np.moveaxis(err, 0, -1)   # ious dim is before the others
                common.plot_lines_with_err(ax, xiou, values, labels, err, values if show_val else None, legend_loc="upper right")
                ax.set_title(f'{vname} for {title}')
                ax.set_xlabel("IoU thresholds")
                ax.set_ylabel("Percentage")
            fig.tight_layout(pad=.2)
            if save_path is not None:
                ext = os.path.splitext(save_path)[1]
                plt.savefig(save_path.replace(ext, f'_{code}{ext}'), dpi=400)

