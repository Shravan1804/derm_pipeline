import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training.infer_images import ImageInference
from classification.train_classification import ImageClassificationTrainer


class ImageClassificationInference(ImageInference):
    @staticmethod
    def get_argparser(parser):
        parser = super(ImageClassificationInference, ImageClassificationInference).get_argparser(parser)
        parser.add_argument('--topk', type=int, default=2, help="Top probas to be written on img")
        return parser

    @staticmethod
    def prepare_inference(args):
        super(ImageClassificationInference, ImageClassificationInference).prepare_inference(args)
        ImageClassificationTrainer.prepare_training(args)

    def maybe_get_labels(self, impath): return None

    def prepare_learner_input(self, inference_item):
        img_path, label = inference_item
        with_labels = label is not None
        if with_labels: raise NotImplementedError
        im_patches, pms = self.maybe_patch(img_path)
        linput = [(p,) for p in im_patches]
        return linput, with_labels, pms

    def process_results(self, inference_item, interp, save_dir):
        img_path, gt = inference_item
        im = common.trainer.load_image_item(img_path)
        pred = interp.preds.topk(self.args.topk, axis=1)
        save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0] + "_preds.jpg")
        if not self.args.no_plot: self.plot_results(interp, im, gt, pred, save_path)
        return pred

    def plot_results(self, interp, im, gt, pred, save_path):
        topk_p, topk_idx = pred
        with_labels = gt is not None
        ncols = 2 if with_labels else 1
        fig, axs = common.prepare_img_axs(im.shape[0] / im.shape[1], 1, ncols, flatten=True)
        if not with_labels: axs = [axs]
        common.img_on_ax(im, axs[0], title='Original image')
        pred_pos = [(0, 0)] if interp.pms is None else [(pm['h'], pm['w'])for pm in interp.pms]
        for (h, w), p, prob in zip(pred_pos, topk_idx, topk_p):
            axs[0].text(w+50, h + 50, f'{self.args.cats[p]}: {prob:.2f}')
        axi = 1
        if with_labels:
            agg_perf = self.trainer.aggregate_test_performance([self.trainer.process_test_preds(interp)])
            self.trainer.plot_custom_metrics(axs[axi], agg_perf, show_val=True)
        if save_path is not None: common.plt_save_fig(save_path, fig=fig, dpi=150)


def main(args):
    classif = ImageClassificationInference(ImageClassificationTrainer(args))
    classif.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    parser = ImageClassificationInference.get_argparser(ImageClassificationTrainer.get_argparser(pdef=pdef))
    args = parser.parse_args()

    ImageClassificationInference.prepare_inference(args)

    common.time_method(main, args)

