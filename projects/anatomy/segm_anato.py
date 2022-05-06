#!/usr/bin/env python

"""segm_anato.py: Train anatomy segmentation model."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

# Run on dgx
# parallel single gpu
# python /workspace/code/derm_pipeline/projects/anatomy/segm_anato.py --encrypted --data /workspace/data/anatomy_project --exp-name anato --logdir /workspace/logs --gpu-ids 0 --deterministic --region hand 2>&1 | tee /workspace/logs/anato.txt

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from segmentation.train_segmentation import ImageSegmentationTrainer


class AnatoSegmTrainer(ImageSegmentationTrainer):
    # dict with body region as key and value a tuple with encrypted data path from root anatomy dir and anatomy cats
    ANATO_DATA_CATS = dict(sorted({
        'ear': (f"ear_anatomy/ear_segm_splitted_encrypted",
                ['non_ear', 'anti_helix', 'anti_tragus', 'concha_cavum', 'concha_cymba', 'ext_auditory_canal',
                 'helical_root', 'helix', 'lobule', 'notch', 'scaphoid_fossa', 'tragus', 'triangular_fossa']),
        'eye': (f"eye_anatomy/eye_segm_splitted_encrypted",
                ['non_eye', 'abnormal', 'conjunctiva', 'eye_margin_lashes_roots', 'eyebrow', 'iris', 'lateral_canthus',
                 'lower_eyelid', 'medial_canthus', 'pupil', 'upper_eyelid']),
        'hand': (f"hands_anatomy/hands_segm_splitted_encrypted",
                 ['non_hand', 'DIP2', 'DIP3', 'DIP4', 'DIP5', 'IP', 'MCP1', 'MCP2', 'MCP3', 'MCP4', 'MCP5', 'PIP2',
                  'PIP3',
                  'PIP4', 'PIP5', 'dorsal_mid', 'dorsal_radial', 'dorsal_ulnar', 'hypothenar', 'index_distal',
                  'index_middle', 'index_proximal', 'little_f_distal', 'little_f_middle', 'little_f_proximal',
                  'middle_f_distal', 'middle_f_middle', 'middle_f_proximal', 'nail', 'palm', 'ring_f_distal',
                  'ring_f_middle', 'ring_f_proximal', 'thenar', 'thumb_distal', 'thumb_proximal', 'wrist']),
        'nail': (f"nail_anatomy/nail_segm_splitted_encrypted",
                 ['non_nail', 'cuticle', 'distal_edge_plate', 'distal_groove', 'hyponychium', 'lateral_fold', 'lunula',
                  'onychodermal_band', 'plate', 'proximal_fold', 'pulp']),
        'mouth': (f"mouth_anatomy/before_correction/mouth_segm_splitted_encrypted",
                  ['non_mouth', 'cutaneous_lip_lower', 'cutaneous_lip_upper', 'inside_mouth', 'oral_commisure', 'teeth',
                   'tongue', 'vermillion_lower', 'vermillion_upper']),
    }.items()))

    @staticmethod
    def get_argparser(desc="Anatomy segmentation trainer arguments", pdef=dict(), phelp=dict()):
        """Creates segmentation argparser
        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(AnatoSegmTrainer, AnatoSegmTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument('--region', required=True, choices=['all', *AnatoSegmTrainer.ANATO_DATA_CATS.keys()],
                            help="Body region to train")
        return parser

    @staticmethod
    def prepare_training(args):
        """Sets up training, check args validity.
        :param args: command line args
        """
        assert args.region in AnatoSegmTrainer.ANATO_DATA_CATS or args.region == 'all'
        args.exp_name = f"{args.region}-anato_" + args.exp_name
        if not args.encrypted:
            AnatoSegmTrainer.ANATO_DATA_CATS = {k: (v1.replace('_encrypted', ''), v2) for k, (v1, v2)
                                                in AnatoSegmTrainer.ANATO_DATA_CATS.items()}
        if args.region == 'all':
            args.sl_train = [os.path.join(d, 'train') for d, _ in AnatoSegmTrainer.ANATO_DATA_CATS.values()]
            args.sl_tests = [os.path.join(d, 'test') for d, _ in AnatoSegmTrainer.ANATO_DATA_CATS.values()]
            args.cats = ["non__" + "_".join([n for n in AnatoSegmTrainer.ANATO_DATA_CATS.keys()]),
                         *[c for _, (_, cs) in AnatoSegmTrainer.ANATO_DATA_CATS.items() for c in cs
                           if not c.startswith('non_')]]
        else:
            args.sl_train = [os.path.join(AnatoSegmTrainer.ANATO_DATA_CATS[args.region][0], 'train')]
            args.sl_tests = [os.path.join(AnatoSegmTrainer.ANATO_DATA_CATS[args.region][0], 'test')]
            args.cats = AnatoSegmTrainer.ANATO_DATA_CATS[args.region][1]
        super(AnatoSegmTrainer, AnatoSegmTrainer).prepare_training(args)

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        self.region_offsets = np.cumsum([0, *[len(cs)-1 for _, cs in list(AnatoSegmTrainer.ANATO_DATA_CATS.values())[:-1]]])
        self.mask_conversion = np.frompyfunc(self.orig_label_to_multi_regions_label, 2, 1)

    def orig_label_to_multi_regions_label(self, label, region_idx):
        return label if label == 0 else label + self.region_offsets[region_idx]

    def load_mask(self, item, load_mask_array=False):
        """Loads image mask
        :param item: str, mask path or array
        :param load_mask_array: bool, optional, force mask array loading in memory
        :return: array mask if encrypted else str mask path
        """
        mask = super().load_mask(item, load_mask_array=True)
        if args.region == 'all':
            region_idx = [i for i, (p, _) in enumerate(AnatoSegmTrainer.ANATO_DATA_CATS.values()) if p in str(item)][0]
            mask = self.mask_conversion(mask, np.ones_like(mask)*region_idx).astype(np.uint8)
        return mask


def main(args):
    """Creates anato segmentation trainer
    :param args: command line args
    """
    anato_segm = AnatoSegmTrainer(args)
    anato_segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 4, '--model': 'resnet50', '--input-size': 380, '--fepochs': 10, '--epochs': 30}
    parser = AnatoSegmTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    AnatoSegmTrainer.prepare_training(args)

    common.time_method(main, args)

