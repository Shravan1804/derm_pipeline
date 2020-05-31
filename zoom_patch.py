import os
import cv2

from preprocessor import ImgClassifPreprocessor
import common
from PatchExtractor import PatchExtractor, get_patcher_arg_parser


class ZoomPatcher(ImgClassifPreprocessor):
    def __init__(self, data, dest, workers=None, zoom_facts=[1], patch_size=512):
        super().__init__(data, dest, workers)
        self.zoom_facts = zoom_facts
        self.patch_size = patch_size

    def get_new_img_sizes(self, h_orig, w_orig):
        facts_new_sizes = []
        for f in self.zoom_facts:
            new_size = (int(w_orig*f), int(h_orig*f))     # cv2 expects (WIDTH, HEIGHT)
            if self.patch_size > new_size[0] or self.patch_size > new_size[1]:
                continue
            else:
                facts_new_sizes.append((f, new_size))
        if not facts_new_sizes:
            smallest = min(h_orig, w_orig)
            ratio = self.patch_size / smallest
            new_size = (max(self.patch_size, int(w_orig * ratio)), max(self.patch_size, int(h_orig * ratio)))
            facts_new_sizes.append((ratio, new_size))
        return facts_new_sizes

    def preprocess(self, pid, pmq, batch):
        print("Proc", pid, "started pre-processing", len(batch), "images.")
        for img_path in batch:
            img_name, ext = os.path.splitext(os.path.basename(img_path))
            dest_dir = os.path.basename(os.path.dirname(img_path))
            orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if len(orig.shape) < 3:
                print("Proc", pid, "skips", img_path, "as its shape is", orig.shape, ".")
                continue
            facts_new_sizes = self.get_new_img_sizes(*orig.shape[:-1])
            patcher = PatchExtractor(patch_size=self.patch_size)
            for fact, new_size in facts_new_sizes:
                im = cv2.resize(orig, new_size)
                zoom_img_name = f'{img_name}_zoom_{round(fact, 2)}{ext}'
                for pm in patcher.patch_grid(zoom_img_name, im_arr=im):
                    cv2.imwrite(os.path.join(self.dest, dest_dir, pm['patch_path']), patcher.pm_to_patch(im, pm))
                    patcher.pm_to_patch(im, pm)
        print("Proc", pid, "completed pre-processing", len(batch), "images.")


def main(args):
    preproc = ZoomPatcher(args.data, args.dest, zoom_facts=args.zoom)
    preproc.apply()


if __name__ == '__main__':
    parser = get_patcher_arg_parser(desc="Augments image dataset with zoom and then patch the images")
    parser.add_argument('--zoom', default=[.5, .75, 1, 1.5, 2], nargs='+', type=float, help="zoom factors")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)
    if args.dest is None:
        args.dest = common.maybe_create(os.path.dirname(args.data),
                                        f'{os.path.basename(args.data)}_zoomed_patched{args.patch_size}')

    common.time_method(main, args)

