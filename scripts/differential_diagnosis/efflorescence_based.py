import os
import sys

import pandas as pd
import fastai.vision.all as fv

from src.general import common
from src.classification.classification_with_image_and_metadata import ImageMetadataClassificationTrainer


class EffsDDTrainer(ImageMetadataClassificationTrainer):
    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        args.exp_name = "effs_dd_" + args.exp_name
        if args.metadata_labels is None:
            args.metadata_labels = os.path.join(args.data, "labels_encrypted.p" if args.encrypted else "labels.p")
        super(EffsDDTrainer, EffsDDTrainer).prepare_training(args)

    def __init__(self, args, **kwargs):
        """Creates trainer
        :param args: command line args
        """
        self.healthy_cat = 'healthy'
        efflorescences = sorted([self.healthy_cat, "erosion", "macule", "papule", "patch", "plaque", "scales"])
        # place here as base_trainer init will create custom metrics which need all cats
        args.cats = sorted([self.healthy_cat, *args.cats])
        super().__init__(args, efflorescences, **kwargs)

    def create_image_to_metadata_dict(self):
        """ Creates image to efflorescence dict
        :return: dict with image name as keys and corresponding list of efflorescences as values
        """
        df_labels = pd.read_pickle(self.args.metadata_labels)
        merge = {'erosion': ['atrophy', 'fissure'], 'scales': ['crust'], 'plaque': ['pustule', 'vesicle']}
        image_to_effs = {}
        for rid, r in df_labels.iterrows():
            effs = [e for e in self.metadata_cats if (e in r and r[e]) or (e in merge and r[merge[e]].any())]
            image_to_effs[os.path.basename(r['imname'])] = [self.healthy_cat] if len(effs) == 0 else effs
        return image_to_effs

    def load_items(self, set_dir):
        """Loads training items from directory. Checks if no effs in which case diagnosis is healthy.
        :param set_dir: str, directory containing training items
        :return: tuple of fastai L lists, first list contain items, second list contains labels
        """
        impaths, labels = super().load_items(set_dir)
        h = [self.healthy_cat]
        return impaths, fv.L([l if self.image_to_metadata(p) != h else h[0] for p, l in zip(impaths, labels)])


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

