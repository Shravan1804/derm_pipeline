#!/usr/bin/env python

"""location_based.py: Train location based differential diagnosis model."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

# Run on dgx
# python /workspace/code/derm_pipeline/training/distributed_launch.py --encrypted /workspace/code/derm_pipeline/projects/differential_diagnosis/location_based.py --data /workspace/data/diff_diags/loc_with_diff_diags/localisation_diff_diags_auto_splitted_encrypted --exp-name loc_based_dd --logdir /workspace/logs --reproducible --gpu-ids --full-precision 0 2>&1 | tee /workspace/logs/loc_dd.txt

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from classification.classification_with_image_and_metadata import ImageMetadataClassificationTrainer


class LocationDDTrainer(ImageMetadataClassificationTrainer):
    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        args.exp_name = "loc_dd_" + args.exp_name
        if args.metadata_labels is None:
            args.metadata_labels = os.path.join(args.data, f'{os.path.basename(args.data)}_preds_df.p')
        super(LocationDDTrainer, LocationDDTrainer).prepare_training(args)

    def __init__(self, args, **kwargs):
        """Creates trainer
        :param args: command line args
        """
        # place here as base_trainer init will create custom metrics which need all cats
        locations = sorted(set("_".join([c.split('__')[-1] for c in args.cats]).split('_')))
        args.cats = [c.split('__')[0] for c in args.cats]
        super().__init__(args, locations, **kwargs)

    def create_image_to_metadata_dict(self):
        """ Creates image to locations dict
        :return: dict with image name as keys and corresponding list of locations as values
        """
        df_labels = pd.read_pickle(self.args.metadata_labels)
        image_to_locs = {}
        for rid, r in df_labels.iterrows():
            dis_predilection_locs = os.path.basename(os.path.dirname(r['impaths'])).split('__')[-1].split('_')
            im_locs = [r['top0']] if r['top0'] in dis_predilection_locs else []
            if r['top1'] in dis_predilection_locs:
                im_locs.append(r['top1'])
            elif len(im_locs) == 0:
                im_locs = dis_predilection_locs
            image_to_locs[os.path.basename(r['impaths'])] = im_locs
        return image_to_locs

    def get_image_cls(self, img_path):
        """Assumes image class is its directory name
        :param img_path: str or Path, image path
        :return: str, class name of image
        """
        return super().get_image_cls(img_path).split('__')[0]


def main(args):
    """Creates location based differential diagnosis trainer and launch training
    :param args: command line args
    """
    loc_dd_trainer = LocationDDTrainer(args)
    loc_dd_trainer.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 32, '--model': 'resnet34', '--input-size': 512, '--fepochs': 10, '--epochs': 30}
    parser = LocationDDTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    LocationDDTrainer.prepare_training(args)

    common.time_method(main, args)
