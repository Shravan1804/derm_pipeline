import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training.infer_images import ImageInference
from segmentation.train_segmentation import ImageSegmentationTrainer


class ImageSegmentationInference(ImageInference):
    pass


def main(args):
    segm = ImageSegmentationInference(ImageSegmentationTrainer(args))
    segm.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256, '--cats': ["other", "pustules", "spots"]}
    args = ImageSegmentationInference.prepare_inference_args(ImageSegmentationTrainer.get_argparser(pdef=pdef))

    ImageSegmentationTrainer.prepare_training(args)

    common.time_method(main, args)

