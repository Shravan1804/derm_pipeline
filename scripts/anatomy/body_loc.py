from src.classification.train_classification import ImageClassificationTrainer
from src.general import common


def main(args):
    """
    Create coarse loc trainer and launch training.

    :param args: command line args
    """
    coarse_loc_trainer = ImageClassificationTrainer(args)
    coarse_loc_trainer.train_model()


if __name__ == "__main__":
    defaults = {
        "--bs": 32,
        "--model": "efficientnet-b2",
        "--input-size": 260,
        "--fepochs": 10,
        "--epochs": 30,
        "--lr": 0.002,
    }
    parser = ImageClassificationTrainer.get_argparser(
        desc="Coarse loc classification", pdef=defaults
    )
    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(project="vm02-body-loc")
        wandb.config.update(args)

    ImageClassificationTrainer.prepare_training(args)
    common.time_method(main, args)
