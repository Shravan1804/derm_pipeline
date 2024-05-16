from src.general import common
from src.segmentation.train_segmentation import ImageSegmentationTrainer


def main(args):
    """
    Create HE segmentation trainer.

    :param args: command line args
    """
    he_segm = ImageSegmentationTrainer(args)
    he_segm.train_model()


if __name__ == "__main__":
    defaults = {
        "--bs": 16,
        "--model": "resnet18",
        "--input-size": 256,
        "--fepochs": 10,
        "--epochs": 30,
        "--cats": ["other", "skin", "eczema"],
    }
    parser = ImageSegmentationTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)
