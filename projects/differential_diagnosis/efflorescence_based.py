#!/usr/bin/env python

"""efflorescence_based.py: Train efflorescence based differential diagnosis model."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

# Run on dgx
# python /workspace/code/derm_pipeline/training/distributed_launch.py --encrypted /workspace/code/derm_pipeline/projects/differential_diagnosis/efflorescence_based.py --data /workspace/data/diff_diags/hands_splitted_encrypted --sl-train train --sl-tests test --exp-name eff_based_dd --logdir /workspace/logs --reproducible 2>&1 | tee /workspace/logs/effs_dd.txt

import os
import sys

import pandas as pd
import fastai.vision.all as fv

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from classification.train_classification import ImageClassificationTrainer
from classification.classification_with_image_and_metadata import ClassifWithMetadata


class EffsDDTrainer(ImageClassificationTrainer):
    @staticmethod
    def get_argparser(desc="Efflorescence based differential diagnosis trainer arguments", pdef=dict(), phelp=dict()):
        """Creates argparser
        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        parser = super(EffsDDTrainer, EffsDDTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--eff-labels', type=str, default="labels_encrypted.p", help="Efflorescence label file")
        parser.add_argument('--image-only', action='store_true', help="Do not use efflorescence in training")
        parser.add_argument('--nembed', type=int, default=64, help="Embedding size for effs. -1 for no embedding")
        return parser

    @staticmethod
    def get_exp_logdir(args, custom=""):
        """Creates experiment log dir
        :param args: command line arguments
        :param custom: custom string to be added in experiment log dirname
        :return: str, experiment log dir
        """
        d = ''
        if args.image_only:
            d += '_only-img'
        else:
            d += f'_embed{args.nembed}' if args.nembed > 0 else f'_no-embed'
        return super(EffsDDTrainer, EffsDDTrainer).get_exp_logdir(args, custom=f'{d}_{custom}')

    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        args.exp_name = "effs_dd_" + args.exp_name
        if not os.path.exists(args.eff_labels):
            args.eff_labels = os.path.join(args.data, args.eff_labels)
            common.check_file_valid(args.eff_labels)
        super(EffsDDTrainer, EffsDDTrainer).prepare_training(args)

    def __init__(self, args, **kwargs):
        """Creates trainer
        :param args: command line args
        """
        super().__init__(args, **kwargs)
        self.healthy_cat = 'healthy'
        self.args.cats = sorted([self.healthy_cat, *self.args.cats])
        self.efflorescences = sorted([self.healthy_cat, "erosion", "macule", "papule", "patch", "plaque", "scales"])
        self.image_to_effs = self.create_image_to_effs()

    def create_image_to_effs(self):
        """ Creates image to efflorescence dict
        :return: dict with image name as keys and corresponding list of efflorescences as values
        """
        df_labels = pd.read_pickle(self.args.eff_labels)
        merge = {'erosion': ['atrophy', 'fissure'], 'scales': ['crust'], 'plaque': ['pustule', 'vesicle']}
        image_to_effs = {}
        for r in df_labels.iterrows():
            effs = [e for e in self.efflorescences if (e in r and r[e]) or (e in merge and r[merge[e]].any())]
            image_to_effs[os.path.basename(r['imname'])] = [self.healthy_cat] if len(effs) == 0 else effs
        return image_to_effs

    def get_effs(self, impath):
        """Retrieve efflorescence of image
        :param impath: str, image path
        :return: list, corresponding efflorescences
        """
        return self.image_to_effs[os.path.basename(impath)]

    def eff1hot(self, effs):
        """1 hot encoding efflorescences
        :param effs: list, efflorescences to encode
        :return: list with 1hot encoding of provided efflorescences
        """
        return [int(e in effs) for e in self.efflorescences]

    def load_items(self, set_dir):
        """Loads training items from directory. Checks if no effs in which case diagnosis is healthy.
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels
        """
        impaths, labels = super().load_items(set_dir)
        h = [self.healthy_cat]
        return impaths, fv.L([l if self.get_effs(p) != h else h[0] for p, l in zip(impaths, labels)])

    def customize_datablock(self):
        """Provides experiment specific kwargs for DataBlock
        :return: dict with argnames and argvalues
        """
        if self.args.image_only:
            return super().custom_datablock_args()
        else:
            return {
                'blocks': (fv.ImageBlock, fv.MultiCategoryBlock(vocab=self.efflorescences),
                           fv.CategoryBlock(vocab=self.args.cats)),
                'n_inp': 2,
                'get_x': (fv.Pipeline([fv.ItemGetter(0), self.load_image_item]),
                          fv.Pipeline([fv.ItemGetter(0), self.get_effs])),
            }

    def create_learner(self, dls):
        """Creates learner with callbacks
        :param dls: train/valid dataloaders
        :return: learner
        """
        if self.args.image_only:
            return super().create_learner(dls)
        else:
            m = ClassifWithMetadata(len(self.efflorescences), getattr(fv, self.args.model), len(self.args.cats))
            learn = m.create_fastai_learner(dls, **self.customize_learner(dls))
            return self.prepare_learner(learn)


def main(args):
    """Creates efflorescence based differential diagnosis trainer and launch training
    :param args: command line args
    """
    effs_dd_trainer = EffsDDTrainer(args)
    effs_dd_trainer.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 8, '--model': 'resnet18', '--input-size': 512, '--fepochs': 10, '--epochs': 30}
    parser = EffsDDTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    EffsDDTrainer.prepare_training(args)

    common.time_method(main, args)

