import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from general.PatchExtractor import PatchExtractor
from training.infer_images import ImageInference
import segmentation.mask_utils as mask_utils
import segmentation.segmentation_utils as segm_utils
from segmentation.train_segmentation import ImageSegmentationTrainer


class ImageSegmentationInference(ImageInference):
    @staticmethod
    def prepare_inference(args):
        super(ImageSegmentationInference, ImageSegmentationInference).prepare_inference(args)
        ImageSegmentationTrainer.prepare_training(args)

    def inference_items(self):
        """Returns list of tuples of images and mask if available"""
        if self.args.impath is not None: return [(Path(self.args.impath), None)]
        elif os.path.exists(os.path.join(self.args.imdir, self.args.img_dir)):
            return [(i, m if os.path.exists(m) else None) for i, m in zip(*self.trainer.load_items(self.args.imdir))]
        else: return [(f, None) for f in common.list_images(self.args.imdir, full_path=True, posix_path=True)]

    def learner_inference(self, learn):
        for img_path, mask_path in self.inference_items():
            with_labels = mask_path is not None
            im_patches, pms = self.maybe_patch(img_path)
            if with_labels:
                mask_patches, _ = self.maybe_patch(mask_path)
                items = list(zip(im_patches, mask_patches))
            else: items = [(p,) for p in im_patches]
            interp = self.infer_items(learn, items, with_labels)
            interp.pms = pms
            self.process_results(img_path, mask_path, interp, with_labels)

    def process_results(self, img_path, mask_path, interp, with_labels):
        if with_labels: im, gt = segm_utils.load_img_and_mask(img_path, mask_path)
        else: im, gt = common.load_rgb_img(img_path), None
        pred = PatchExtractor.rebuild_im_from_patches(interp.pms, interp.decoded, im.shape[:2])
        save_tag = "_" + os.path.basename(self.args.mdir) if self.args.mdir is not None else ""
        if self.args.ps is not None: save_tag += f'_ps{self.args.ps}px'
        spath = common.maybe_create(self.args.exp_logdir, f'preds{save_tag}_{self.args.exp_name}')
        im_fname = os.path.basename(img_path)
        if not self.args.no_plot: self.plot_results(interp, im_fname, im, gt, pred, with_labels, spath)

    def plot_results(self, interp, im_name, im, gt, pred, with_labels, save_dir):
        ncols = 5 if with_labels else 2
        fig, axs = common.prepare_img_axs(im.shape[0] / im.shape[1], 1, ncols, flatten=True)
        common.img_on_ax(im, axs[0], title='Original image')
        axi = 1
        if with_labels:
            mask_utils.im_mask_on_ax(axs[axi], im, gt, "Ground truth")
            axi += 1
        mask_utils.im_mask_on_ax(axs[axi], im, pred, "Predictions")
        axi += 1
        if with_labels:
            mask_utils.im_mask_on_ax(axs[axi], im, (gt != pred).astype(np.uint8), "Predictions different from GT")
            axi += 1
            agg_perf = self.trainer.aggregate_test_performance([self.trainer.process_test_preds(interp)])
            self.trainer.plot_custom_metrics(axs[axi], agg_perf, show_val=True)
        fig.tight_layout(pad=.2)
        fig.savefig(os.path.join(save_dir, os.path.splitext(im_name)[0]+"_preds.jpg"), dpi=150)


def main(args):
    segm = ImageSegmentationInference(ImageSegmentationTrainer(args))
    segm.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationInference.get_argparser(ImageSegmentationTrainer.get_argparser(pdef=pdef))
    args = parser.parse_args()

    ImageSegmentationInference.prepare_inference(args)

    common.time_method(main, args)

