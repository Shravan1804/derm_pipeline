import os
import cv2
import math
import argparse
import multiprocessing as mp

from scipy import ndimage

import concurrency
import common
from PatchExtractor import PatchExtractor


class ImgPreprocessor:
    def __init__(self, source, dest, workers=None):
        self.source = source
        self.dest = dest
        self.prepare_dest()
        self.workers = mp.cpu_count() - 2 if workers is None else workers
        self.items = self.collect_items()
        self.batches = self.create_batches()

    def apply(self, transforms):
        pmq, jobs = mp.Queue(), []
        # for i, batch in zip(np.resize(range(self.workers), len(self.batches)), self.batches):
        for i, batch in zip(range(self.workers), self.batches):
            jobs.append(mp.Process(target=self.preprocess, args=(i, pmq, batch, transforms)))
            jobs[i].start()
        pms = concurrency.unload_mpqueue(pmq, jobs)
        for j in jobs:
            j.join()
        return pms

    def create_batches(self):
        """Divides items equally among workers. Number of batches = number of workers"""
        batch_size = math.ceil(len(self.items) / self.workers)
        return common.batch_list(self.items, batch_size)

    def collect_items(self):
        return common.list_files_in_dirs(self.source, full_path=True)

    def prepare_dest(self):
        for d in common.list_dirs(self.source):
            path = os.path.join(self.dest, d)
            if not os.path.exists(path):
                os.mkdir(path)

    def preprocess(self, pid, pmq, batch, transforms):
        print("Process", pid, "started pre-processing", len(batch), "images.")
        for img_path in batch:
            dest_dir = os.path.basename(os.path.dirname(img_path))
            orig = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            processed = [(os.path.basename(img_path), orig, True)]  # (orig name, orig img, force apply)
            for transform in transforms:
                processed = transform.apply(processed)
            for name, processed_img in processed:
                cv2.imwrite(os.path.join(self.dest, dest_dir, name), processed_img)
        print("Process", pid, "completed pre-processing", len(batch), "images.")


class ImgTransform:
    def __init__(self, name, force_fact=None):
        self.name = name
        self.force_fact = force_fact

    def apply(self, to_process):
        processed = []
        for name, img, force in to_process:
            if not self.can_apply(img) and force:
                processed.extend(self.force_process(name, img))
            elif self.can_apply(img):
                processed.extend(self.process(name, img))
        assert len(processed) > 0, f"Error, transform {self.name} did not produce output."
        return processed

    def process(self, name, img):
        raise NotImplementedError

    def force_process(self, orig_name, orig_img):
        return self.process(orig_name, orig_img)

    def can_apply(self, img):
        raise NotImplementedError

    def processed_name(self, name, param):
        img_name, ext = os.path.splitext(name)
        return f'{img_name}_{self.name}_{param}{ext}'


class Zoom(ImgTransform):
    def __init__(self, zoom_facts=None):
        super().__init__('zoomed', 1.0)
        self.zoom_facts = [self.force_fact] if zoom_facts is None else zoom_facts

    def can_apply(self, img):
        return True

    def process(self, name, img):
        h, w = img.shape[:2]
        for f in self.zoom_facts:
            new_size = (int(w * f), int(h * f))     # cv2 expects (WIDTH, HEIGHT)
            yield self.processed_name(name, round(f, 2)), cv2.resize(img, new_size), f == self.force_fact


class Rotate(ImgTransform):
    def __init__(self, rot_facts=None, rot_pad='reflect'):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.rotate.html#scipy.ndimage.rotate
        # rotation_padding=‘reflect’ or  ‘constant’ or ‘nearest’ or ‘mirror’ or ‘wrap’
        #    mode       |   Ext   |         Input          |   Ext
        # -----------+---------+------------------------+---------
        # 'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
        # 'reflect'  | 3  2  1 | 1  2  3  4  5  6  7  8 | 8  7  6
        # 'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
        # 'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0
        # 'wrap'     | 6  7  8 | 1  2  3  4  5  6  7  8 | 1  2  3
        super().__init__('rotated', 0)
        self.rot_facts = [self.force_fact] if rot_facts is None else rot_facts
        self.rot_pad = rot_pad

    def can_apply(self, img):
        return True

    def process(self, name, img):
        for r in self.rot_facts:
            yield self.processed_name(name, r), ndimage.rotate(img, r, reshape=True, mode=self.rot_pad),\
                  r == self.force_fact


class Patch(ImgTransform):
    def __init__(self, patch_size=512):
        super().__init__('patched')
        self.patch_size = patch_size
        self.patcher = PatchExtractor(patch_size=self.patch_size)

    def can_apply(self, img):
        h, w = img.shape[:2]
        return h >= self.patch_size and w >= self.patch_size

    def process(self, name, img):
        for pm in self.patcher.patch_grid(name, im_arr=img):
            param = 'h' + str(pm['h']) + '_w' + str(pm['w'])
            yield self.processed_name(name, param), self.patcher.extract_patch(img, pm), True


def main(args):
    preproc = ImgPreprocessor(source=args.data, dest=args.dest, workers=args.workers)
    transforms = [Zoom(args.zoom), Rotate(args.rotate)]
    if not args.no_patch:
        transforms.append(Patch(args.patch_size))

    preproc.apply(transforms)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Augments image dataset with zoom and then patch the images")
    parser.add_argument('--data', type=str, required=True, help="source data root directory absolute path")
    parser.add_argument('--dest', type=str, help="directory where the patches should be saved")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="patch size")
    parser.add_argument('--zoom', default=[.5, .75, 1, 1.5, 2], nargs='+', type=float, help="zoom factors")
    parser.add_argument('--rotate', default=[0, 60, 120, 180, 240, 300], nargs='+', type=float, help="rotate factors")
    parser.add_argument('--no-patch', action='store_true', help="Do not divide images in patch")
    parser.add_argument('--workers', type=int, help="Number of process")
    args = parser.parse_args()

    args.data = args.data.rstrip('/')
    common.check_dir_valid(args.data)
    if args.dest is None:
        suffix = 'zoomed' if args.zoom_only else f'zoomed_patched{args.patch_size}'
        args.dest = common.maybe_create(os.path.dirname(args.data), f'{os.path.basename(args.data)}_{suffix}')

    common.time_method(main, args)

