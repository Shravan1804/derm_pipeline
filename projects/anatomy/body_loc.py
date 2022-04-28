#!/usr/bin/env python

"""body_loc.py: Train coarse localization model."""

__author__ = "Ludovic Amruthalingam"
__maintainer__ = "Ludovic Amruthalingam"
__email__ = "ludovic.amruthalingam@unibas.ch"
__status__ = "Development"
__copyright__ = (
    "Copyright 2021, University of Basel",
    "Copyright 2021, Lucerne University of Applied Sciences and Arts"
)

# Run on dgx
# python /workspace/code/derm_pipeline/training/distributed_launch.py --encrypted /workspace/code/derm_pipeline/projects/anatomy/body_loc.py --data /workspace/data/anatomy_project/body_loc/USZ_pipeline_cropped_images_patched_512_encrypted --sl-train strong_labels_train --sl-tests strong_labels_test_balanced510 --progr-size --exp-name body_loc --logdir /workspace/logs --reproducible 2>&1 | tee /workspace/logs/body_loc.txt


import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from classification.train_classification import ImageClassificationTrainer


def main(args):
    """Creates coarse loc trainer and launch training
    :param args: command line args
    """
    coarse_loc_trainer = ImageClassificationTrainer(args)
    coarse_loc_trainer.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 32, '--model': 'efficientnet-b2', '--input-size': 260, '--fepochs': 10, '--epochs': 30}
    parser = ImageClassificationTrainer.get_argparser(desc="Coarse loc classification", pdef=defaults)
    args = parser.parse_args()

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args)

