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
# python /workspace/code/derm_pipeline/training/distributed_launch.py --encrypted /workspace/code/derm_pipeline/projects/anatomy/efflorescence_based.py --data /workspace/data/diff_diags/hands_splitted_encrypted --sl-train train --sl-tests test --exp-name eff_based_dd --logdir /workspace/logs --reproducible 2>&1 | tee /workspace/logs/effs_dd.txt

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, os.path.pardir)))
from general import common
from classification.train_classification import ImageClassificationTrainer


def main(args):
    """Creates efflorescence based differential diagnosis trainer and launch training
    :param args: command line args
    """
    effs_dd_trainer = ImageClassificationTrainer(args)
    effs_dd_trainer.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 8, '--model': 'resnet18', '--input-size': 512, '--fepochs': 10, '--epochs': 30}
    parser = ImageClassificationTrainer.get_argparser(desc="Efflorescence based differential diagnosis", pdef=defaults)
    args = parser.parse_args()

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args)

