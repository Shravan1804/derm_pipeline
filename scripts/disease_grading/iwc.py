from src.general import common
from src.segmentation.train_segmentation import ImageSegmentationTrainer


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

