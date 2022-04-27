

#/workspace/data/ppp_grading/ppp_study_segm_splitted_encrypted

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from projects.disease_grading.ppp import PPPTrainer


class PPPStudyTrainer(PPPTrainer):
    def load_mask(self, item, load_mask_array=False):
        # ['not_skin', 'skin', 'scales', 'brown_spot-like', 'pustule-like', 'brown_spot', 'pustule']
        mask = super().load_mask(item, load_mask_array=True)
        mask[mask < 5] = 0
        mask[mask == 5] = 2
        mask[mask == 6] = 1
        # ["other", "pustules", "spots"]
        return mask


def main(args):
    """Creates a ppp segmentation trainer
    :param args: command line args
    """
    ppp_trainer = PPPStudyTrainer(args)
    ppp_trainer.train_model()


if __name__ == '__main__':
    defaults = {'--bs': 16, '--model': 'resnet18', '--input-size': 380, '--fepochs': 10, '--epochs': 30}
    parser = PPPStudyTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    PPPStudyTrainer.prepare_training(args)

    common.time_method(main, args)
