#!/usr/bin/env python

"""ppp.py: Train hand iwc model."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

# Run on dgx
# distributed
# python TODO
# parallel single gpu
# python /workspace/code/derm_pipeline/projects/disease_grading/iwc.py --encrypted --data /workspace/data/disease_grading/iwc/segm_iwc_with_objs_splitted_patched_512_encrypted --exp-name iwc --logdir /workspace/logs --gpu-ids 0 --reproducible 2>&1 | tee /workspace/logs/iwc.txt

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from segmentation.train_segmentation import ImageSegmentationTrainer


def main(args):
    """Creates IWC segmentation trainer
    :param args: command line args
    """
    iwc_segm = ImageSegmentationTrainer(args)
    iwc_segm.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 16, '--model': 'resnet18', '--input-size': 256, '--fepochs': 10, '--epochs': 30,
                '--cats': ["other", "white_skin", "non_white_skin"]}
    parser = ImageSegmentationTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)

