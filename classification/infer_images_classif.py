import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir)))
from general import common
from training.infer_images import ImageInference
from classification.train_classification import ImageClassificationTrainer


class ImageClassificationInference(ImageInference):
    pass


def main(args):
    classif = ImageClassificationInference(ImageClassificationTrainer(args))
    classif.inference()


if __name__ == '__main__':
    pdef = {'--bs': 6, '--model': 'resnet34', '--input-size': 256}
    args = ImageClassificationInference.prepare_inference_args(ImageClassificationTrainer.get_argparser(pdef=pdef))

    ImageClassificationTrainer.prepare_training(args)

    common.time_method(main, args)

