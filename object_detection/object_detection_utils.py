
import numpy as np
from pycocotools.cocoeval import COCOeval


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
        min_dets = np.max(dets)
        return [min_dets, *tuple(int(f*min_dets) for f in facts)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cats = ["all"] + [self.cocoGt.cats[cid]['name'] for cid in self.params.catIds]
        self.ious_summary = [.15, .25, .5, .75]
        self.params.iouThrs = np.linspace(.15, 0.95, np.int(np.round((0.95 - .15) / .05)) + 1, endpoint=True)
        self.params.areaRng = self.computeAreaRng()
        self.params.maxDets = self.computeMaxDets()

    def eval_acc_summarize(self):
        self.evaluate()
        self.accumulate()
        self.summarize()

    def summarize(self):
        if not self.eval:
            raise Exception('Please run accumulate() first')

        p = self.params

        def _iou_res(iouThr, s):
            return s if iouThr is not None else s[iouThr == p.iouThrs]

        def _summarize(iouThr=None, areaRng=p.areaRngLbl[0], maxDets=p.maxDets[0], cat=self.cats[0]):
            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            cind = [i for i, c in enumerate(p.catIds) if self.cats[c] == cat or cat == self.cats[0]]

            # dimension of precision: [TxRxKxAxM]
            sap = _iou_res(iouThr, self.eval['precision'])[:, :, cind, aind, mind]
            mean_sap = -1 if not len(sap[sap > -1]) else np.mean(sap[sap > -1])

            # dimension of recall: [TxKxAxM]
            sar = _iou_res(iouThr, self.eval['recall'])[:, cind, aind, mind]
            mean_sar = -1 if not len(sar[sar > -1]) else np.mean(sar[sar > -1])

            return mean_sap, mean_sar

        stats = []
        for cat in self.cats:
            print("CATEGORY:", cat)
            for area in p.areaRngLbl:
                print("\tOBJECT SIZE:", area)
                for det in p.maxDets:
                    print("\t\tMAX DET COUNT:", det)
                    for iou in self.ious_summary + [None]:
                        iouStr = f'{p.iouThrs[0]:0.2f}:{p.iouThrs[-1]:0.2f}' if iou is None else f'{iou:0.2f}'
                        ap, ar = _summarize(iou, area, det, cat)
                        print(f"\t\t\tIOU={iouStr}: AP={ap:0.3f}, AR={ar:0.3f}")
                        stats.append((ap, ar))
        self.stats = np.array(stats)

