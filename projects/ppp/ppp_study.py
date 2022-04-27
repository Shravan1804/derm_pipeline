import os
import sys

import numpy as np
from p_tqdm import p_umap

import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from segmentation.train_segmentation import ImageSegmentationTrainer


class TrainPPPStudy(ImageSegmentationTrainer):
    def load_mask(self, item, load_mask_array=False):
        # ['not_skin', 'skin', 'scales', 'brown_spot-like', 'pustule-like', 'brown_spot', 'pustule']
        mask = super().load_mask(item, load_mask_array=True)
        mask[mask < 5] = 0
        mask[mask == 5] = 2
        mask[mask == 6] = 1
        # ["other", "pustules", "spots"]
        return mask

    def item_with_lesions(self, train_item):
        impath, mpath = train_item
        mask = self.load_mask(mpath)
        return 1 in mask or 2 in mask

    def get_train_items(self, merged=True):
        sl_data, wl_data = super().get_train_items(merged=merged)
        print("#train items:", len(sl_data[0]))
        zipped_sl_data = sl_data.zip()
        filtered = p_umap(self.item_with_lesions, zipped_sl_data, num_cpus=1 / len(self.args.gpu_ids))
        sl_data = fv.L([(i, m) for (i, m), keep in zip(zipped_sl_data, filtered) if keep]).zip()
        print("#train items with lesions:", len(sl_data[0]))
        return sl_data, wl_data


def main(args):
    """Creates segmentation trainer
    :param args: command line args
    """
    segm = ImageSegmentationTrainer(args)
    segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 8, '--model': 'resnet50', '--input-size': 380, '--cats': ["other", "pustules", "spots"],
                '--epochs': 42}
    parser = ImageSegmentationTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args(['--data', "/workspace/data/ppp_grading/ppp_study_segm_splitted_encrypted",
                              '--sl-train', 'train', '--sl-tests', 'test', '--fepochs', '16',
                              '--exp-name', 'ppp_study', '--full-precision', '--focal-loss-plus-dice-focal-loss',
                              '--encrypted'])

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)
