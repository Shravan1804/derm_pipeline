import os
import sys

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
    def get_argparser(parser):
        parser = super(ImageSegmentationInference, ImageSegmentationInference).get_argparser(parser)
        parser.add_argument('--gt-pred-diff', action='store_true', help="Creates Preds != GT graph")
        return parser

    @staticmethod
    def prepare_inference(args):
        super(ImageSegmentationInference, ImageSegmentationInference).prepare_inference(args)
        ImageSegmentationTrainer.prepare_training(args)

    def maybe_get_labels(self, impath):
        return self.trainer.get_image_mask_path(impath)

    def prepare_learner_input(self, inference_item):
        img_path, mask_path = inference_item
        with_labels = mask_path is not None
        im_patches, pms = self.maybe_patch(img_path)
        linput = list(zip(im_patches, self.maybe_patch(mask_path)[0])) if with_labels else [(p,) for p in im_patches]
        return linput, with_labels, pms

    def process_results(self, inference_item, interp, save_dir):
        img_path, mask_path = inference_item
        im = common.trainer.load_image_item(img_path)
        gt = common.trainer.load_mask(mask_path) if mask_path is not None else None
        if interp.pms is None: pred = mask_utils.resize_mask(interp.decoded[0].numpy(), im.shape[:2])
        else: pred = PatchExtractor.rebuild_im_from_patches(interp.pms, interp.decoded.numpy(), im.shape[:2])
        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + "_preds.jpg")
        if not self.args.no_plot: self.plot_results(interp, im, gt, pred, save_path)
        return pred

    def plot_results(self, interp, im, gt, pred, save_path):
        with_labels = gt is not None
        if with_labels: ncols = 5 if self.args.gt_pred_diff else 4
        else: ncols = 2
        fig, axs = common.prepare_img_axs(im.shape[0] / im.shape[1], 1, ncols, flatten=True)
        common.img_on_ax(im, axs[0], title='Original image')
        axi = 1
        if with_labels:
            mask_utils.im_mask_on_ax(axs[axi], im, gt, "Ground truth")
            axi += 1
        mask_utils.im_mask_on_ax(axs[axi], im, pred, "Predictions")
        axi += 1
        if with_labels:
            if self.args.gt_pred_diff:
                mask_utils.im_mask_on_ax(axs[axi], im, (gt != pred).astype(np.uint8), "Preds != GT")
                axi += 1
            agg_perf = self.trainer.aggregate_test_performance([self.trainer.process_test_preds(interp)])
            self.trainer.plot_custom_metrics(axs[axi], agg_perf, show_val=True)
        if save_path is not None: common.plt_save_fig(save_path, fig=fig, dpi=150)


def main(args):
    segm = ImageSegmentationInference(ImageSegmentationTrainer(args))
    segm.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    parser = ImageSegmentationInference.get_argparser(ImageSegmentationTrainer.get_argparser(pdef=pdef))
    args = parser.parse_args()

    ImageSegmentationInference.prepare_inference(args)

    common.time_method(main, args)

