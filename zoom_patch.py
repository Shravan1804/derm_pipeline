import os
import cv2

from preprocessor import ImgClassifPreprocessor
import common
from PatchExtractor import PatchExtractor, get_patcher_arg_parser


class Zoom(ImgClassifPreprocessor):
    def __init__(self, source, dest, workers=None, zoom_facts=[1], mask_dir_prefix='masks_'):
        super().__init__(source, dest, workers)
        self.zoom_facts = zoom_facts
        self.mask_dir_prefix = mask_dir_prefix

    def prepare_orig_img(self, img_path):
        orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        dest_dir = os.path.basename(os.path.dirname(img_path))
        skip = len(orig.shape) < 3 and not dest_dir.startswith(self.mask_dir_prefix)
        if skip:
            print("Skipping", img_path, "as its shape is", orig.shape, ".")
        return skip, orig, dest_dir

    def get_new_img_sizes(self, h_orig, w_orig):
        for f in self.zoom_facts:
            yield f, (int(w_orig * f), int(h_orig * f))     # cv2 expects (WIDTH, HEIGHT)

    def img_zooms(self, img_path, orig):
        for fact, new_size in self.get_new_img_sizes(*orig.shape[:2]):
            img_name, ext = os.path.splitext(os.path.basename(img_path))
            yield f'{img_name}_zoom_{round(fact, 2)}{ext}', cv2.resize(orig, new_size)

    def preprocess(self, pid, pmq, batch):
        print("Proc", pid, "started zoom pre-processing", len(batch), "images.")
        for img_path in batch:
            skip, orig, dest_dir = self.prepare_orig_img(img_path)
            if not skip:
                for zoom_img_name, im in self.img_zooms(img_path, orig):
                    cv2.imwrite(os.path.join(self.dest, dest_dir, zoom_img_name), im)
        print("Proc", pid, "completed zoom pre-processing", len(batch), "images.")


class ZoomPatcher(Zoom):
    def __init__(self, source, dest, workers=None, zoom_facts=[1], mask_dir_prefix='masks_', patch_size=512):
        super().__init__(source, dest, workers, zoom_facts, mask_dir_prefix)
        self.patch_size = patch_size

    def get_new_img_sizes(self, h_orig, w_orig):
        max_f = max(self.zoom_facts)
        if max_f * h_orig < self.patch_size or max_f * w_orig < self.patch_size:
            smallest = min(h_orig, w_orig)
            ratio = self.patch_size / smallest
            new_size = (max(self.patch_size, int(w_orig * ratio)), max(self.patch_size, int(h_orig * ratio)))
            yield ratio, new_size
        else:
            for f, new_size in super().get_new_img_sizes(h_orig, w_orig):
                if self.patch_size <= new_size[0] and self.patch_size <= new_size[1]:
                    yield f, new_size

    def preprocess(self, pid, pmq, batch):
        print("Proc", pid, "started pre-processing", len(batch), "images.")
        for img_path in batch:
            skip, orig, dest_dir = self.prepare_orig_img(img_path)
            if not skip:
                patcher = PatchExtractor(patch_size=self.patch_size)
                for zoom_img_name, im in self.img_zooms(img_path, orig):
                    for pm in patcher.patch_grid(zoom_img_name, im_arr=im):
                        cv2.imwrite(os.path.join(self.dest, dest_dir, pm['patch_path']), patcher.pm_to_patch(im, pm))
        print("Proc", pid, "completed pre-processing", len(batch), "images.")


def main(args):
    params = {'source': args.data, 'dest': args.dest, 'workers': args.workers, 'zoom_facts': args.zoom}
    if args.zoom_only:
        preproc = Zoom(**params)
    else:
        params['patch_size'] = args.patch_size
        preproc = ZoomPatcher(**params)

    preproc.apply()


if __name__ == '__main__':
    parser = get_patcher_arg_parser(desc="Augments image dataset with zoom and then patch the images")
    parser.add_argument('--zoom', default=[.5, .75, 1, 1.5, 2], nargs='+', type=float, help="zoom factors")
    parser.add_argument('--zoom-only', action='store_true', help="Applies only zoom augmentation")
    parser.add_argument('--workers', type=int, help="Number of process")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)
    if args.dest is None:
        suffix = 'zoomed' if args.zoom_only else f'zoomed_patched{args.patch_size}'
        args.dest = common.maybe_create(os.path.dirname(args.data), f'{os.path.basename(args.data)}_{suffix}')

    common.time_method(main, args)

