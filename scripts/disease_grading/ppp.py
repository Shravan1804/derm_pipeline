import fastai.vision.all as fv

from src.general import common
from src.segmentation.train_segmentation import ImageSegmentationTrainer


class PPPTrainer(ImageSegmentationTrainer):
    @staticmethod
    def get_argparser(
        desc="PPP segmentation trainer arguments", pdef=dict(), phelp=dict()
    ):
        """
        Create segmentation argparser.

        :param desc: str, argparser description
        :param pdef: dict, default arguments values
        :param phelp: dict, argument help strings
        :return: argparser
        """
        # static method super call: https://stackoverflow.com/questions/26788214/super-and-staticmethod-interaction
        parser = super(PPPTrainer, PPPTrainer).get_argparser(desc, pdef, phelp)
        parser.add_argument(
            "--include-skin", action="store_true", help="Model will also segment skin"
        )
        parser.add_argument(
            "--only-with-lesions",
            action="store_true",
            help="During training, consider only patches" "with lesions",
        )
        return parser

    @staticmethod
    def prepare_training(args):
        """
        Set up training, check args validity.

        :param args: command line args
        """
        args.exp_name = (
            "ppp_" + ("with_skin_" if args.include_skin else "") + args.exp_name
        )
        args.cats = (
            ["other", "skin", "pustules", "spots"]
            if args.include_skin
            else ["other", "pustules", "spots"]
        )
        super(PPPTrainer, PPPTrainer).prepare_training(args)

    def load_mask(self, item, load_mask_array=False):
        """
        Load image mask.

        :param item: str, mask path or array
        :param load_mask_array: bool, optional, force mask array loading in memory
        :return: array mask if encrypted else str mask path
        """
        # Original mask classes: ['other', 'skin', 'pustules', 'spots'']
        mask = super().load_mask(item, load_mask_array=True)
        if not self.args.include_skin:
            mask[mask == 1] = 0
            mask[mask > 1] -= 1
        return mask

    def item_with_lesions(self, train_item):
        impath, mpath = train_item
        mask = self.load_mask(mpath, load_mask_array=True)
        return (
            (2 in mask or 3 in mask)
            if self.args.include_skin
            else (1 in mask or 2 in mask)
        )

    def get_train_items(self, merged=True):
        sl_data, wl_data = super().get_train_items(merged=merged)
        if self.args.only_with_lesions:
            print("#train items:", len(sl_data[0]))
            sl_data = fv.L(
                [
                    train_item
                    for train_item in sl_data.zip()
                    if self.item_with_lesions(train_item)
                ]
            ).zip()
            print("#train items with lesions:", len(sl_data[0]))
        return sl_data, wl_data


def main(args):
    """
    Create a ppp segmentation trainer.

    :param args: command line args
    """
    ppp_trainer = PPPTrainer(args)
    ppp_trainer.train_model()


if __name__ == "__main__":
    defaults = {
        "--bs": 8,
        "--model": "resnet50",
        "--input-size": 380,
        "--fepochs": 10,
        "--epochs": 30,
    }
    parser = PPPTrainer.get_argparser(pdef=defaults)
    args = parser.parse_args()

    if args.wandb:
        import wandb

        wandb.init(project="vm02-PPP")
        wandb.config.update(args)

    PPPTrainer.prepare_training(args)
    common.time_method(main, args)
