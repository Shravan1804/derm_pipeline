# DGX
# parallel single gpu
# python /workspace/code/derm_pipeline/projects/disease_grading/ppp_study.py --encrypted --data /workspace/data/disease_grading/ppp_grading/ppp_study_segm_splitted_encrypted --exp-name ppp_study --logdir /workspace/logs --gpu-ids 0 --reproducible 2>&1 | tee /workspace/logs/ppp_study.txt

from scripts.disease_grading.ppp import PPPTrainer
from src.general import common


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
    """
    Create a ppp segmentation trainer.

    :param args: command line args
    """
    ppp_trainer = PPPStudyTrainer(args)
    ppp_trainer.train_model()


if __name__ == "__main__":
    defaults = {
        "--bs": 16,
        "--model": "resnet18",
        "--input-size": 380,
        "--fepochs": 10,
        "--epochs": 30,
    }
    parser = PPPStudyTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    PPPStudyTrainer.prepare_training(args)

    common.time_method(main, args)
