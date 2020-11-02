import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
import fastai.vision.all as fv

import common
from PatchExtractor import PatchExtractor
from infer_images import FastaiModel

epsilon = 1e-8

def get_TP_TN_FP_FN(truth, preds):
    TP = (preds & truth).sum().item()
    TN = (~preds & ~truth).sum().item()
    FP = (preds & ~truth).sum().item()
    FN = (~preds & truth).sum().item()
    return TP, TN, FP, FN

def acc(TP, TN, FP, FN):
    return (TP + TN)/(TP + TN + FP + FN + epsilon)

def prec(TP, TN, FP, FN):
    return TP/(TP + FP + epsilon)

def rec(TP, TN, FP, FN):
    return TP/(TP + FN + epsilon)

def prepare_truth_preds(inp, targ, cls_idx, bg, axis):
    inp = inp.argmax(dim=axis)
    if bg is not None:
        mask = targ != bg
        inp, targ = inp[mask], targ[mask]
    truth = targ == cls_idx
    preds = inp == cls_idx
    return truth, preds

def cls_perf(perf, inp, targ, cls_idx, bg=None, axis=1):
    """If bg sets then computes perf without background"""
    truth, preds = prepare_truth_preds(inp, targ, cls_idx, bg, axis)
    return torch.tensor(perf(*get_TP_TN_FP_FN(truth, preds)))

def O_acc(inp, targ): return cls_perf(acc, inp, targ, 0)
def P_acc(inp, targ): return cls_perf(acc, inp, targ, 1, bg=0)
def S_acc(inp, targ): return cls_perf(acc, inp, targ, 2, bg=0)

def O_prec(inp, targ): return cls_perf(prec, inp, targ, 0)
def P_prec(inp, targ): return cls_perf(prec, inp, targ, 1, bg=0)
def S_prec(inp, targ): return cls_perf(prec, inp, targ, 2, bg=0)

def O_rec(inp, targ): return cls_perf(rec, inp, targ, 0)
def P_rec(inp, targ): return cls_perf(rec, inp, targ, 1, bg=0)
def S_rec(inp, targ): return cls_perf(rec, inp, targ, 2, bg=0)

def get_ppp_mask(img_path):
    return fv.Path(str(img_path).replace(f'/images{suf}/', f'/masks{suf}/').replace('.jpg', '.png'))
def get_ppp_images(source):
    return fv.get_image_files(source/"train"/f"images{suf}") + get_image_files(source/"test"/f"images{suf}")
def get_ppp_splitter():
    return fv.GrandparentSplitter(train_name='train', valid_name='test')



class SegmModel(FastaiModel):
    def __init__(self, model_path, output_dir, use_cpu, bs, cats):
        super().__init__(model_path, use_cpu, bs)
        self.output_dir = output_dir
        self.cats = cats


def show_segm_preds(img_arr, pred_mask, gt_mask=None, save_path=None):
    ncols = 2 if gt_mask is None else 3
    _, axs = common.prepare_img_axs(img_arr.shape[0] / img_arr.shape[1], 1, ncols)
    common.img_on_ax(img_arr, axs[0], title='Original image')
    for ax, m, label in zip(axs[1:], [pred_mask, gt_mask], ["Prediction", "Ground truth"]):
        common.img_on_ax(img_arr, ax, title=label)
        ax.imshow(m, cmap='jet', alpha=0.4)
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def create_pred_mask(im_shape, pms, preds):
    mask = np.zeros(im_shape[:2], dtype=np.uint8)
    preds = torch.cat([p for p in preds], dim=0)
    for pm, pred in zip(pms, preds):
        ps, h, w = pm['ps'], pm['h'], pm['w']
        # background preds (and preds smaller than max) are overwritten in case of patch overlap
        mask[h:h+ps, w:w+ps] = np.maximum(pred.numpy(), mask[h:h+ps, w:w+ps])
    return mask


def main(args):
    img_list = [args.img] if args.img is not None else common.list_files(args.img_dir, full_path=True)

    model = SegmModel(args.model, args.out_dir, args.cpu, args.bs, args.cats)

    patcher = PatchExtractor(args.ps)
    for img_id, img_path in enumerate(img_list):
        print(f"Image {img_id+1}/{len(img_list)}: {img_path}")
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = common.load_rgb_img(img_path)

        patch_maps = patcher.patch_grid(img_path, im)
        b_pms = common.batch_list(patch_maps, args.bs)

        preds = []
        for batch in tqdm(b_pms) if len(b_pms) > 1 else b_pms:
            patches = [PatchExtractor.extract_patch(im, pm) for pm in batch]
            batch_preds = model.predict_imgs(patches)
            preds.append(batch_preds)

        pred_mask = create_pred_mask(im.shape, patch_maps, preds)
        if args.show_gt:
            gt_path = os.path.join(args.gt_dir, file +'.png') if args.img_gt is None else args.img_gt
            gt_mask = patcher.load_image(gt_path)
        else:
            gt_mask = None

        if not args.no_graphs:
            save_path = None if args.out_dir is None else os.path.join(args.out_dir, f'{file}_segm.jpg')
            show_segm_preds(im, pred_mask, gt_mask, save_path=save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Apply model to images")
    parser.add_argument('--img-dir', type=str, help="source image root directory absolute path")
    parser.add_argument('--img', type=str, help="absolute path, single image use, will take precedence over img_dir")
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--bs', default=3, type=int, help="batch size")
    parser.add_argument('--ps', default=512, type=int, help="patch size")
    parser.add_argument('--cats', type=str, nargs='+', default=["other", "pustules", "spots"], help="Segm categories")
    parser.add_argument('--out-dir', type=str, help="if save-out set, output dir absolute path")
    parser.add_argument('--no-graphs', action='store_true', help="Do not create graphs")
    parser.add_argument('--show-gt', action='store_true', help="Show ground truth")
    parser.add_argument('--img-gt', type=str, help="Ground truth for provided img")
    parser.add_argument('--gt-dir', type=str, help="Directory of ground truth masks")
    parser.add_argument('--cpu', action='store_true', help="Use CPU (defaults use cuda if available)")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    common.add_multi_gpus_args(parser)
    args = parser.parse_args()

    if args.img is not None:
        common.check_file_valid(args.img)
        if args.show_gt:
            if args.img_gt is None:
                args.img_gt = args.img.replace(f'/images/', f'/masks/').replace('.jpg', '.png')
            common.check_file_valid(args.img_gt)
    else:
        common.check_dir_valid(args.img_dir)
        if args.show_gt:
            if args.gt_dir is None:
                args.gt_dir = args.img_dir.replace(f'/images', f'/masks')
            common.check_dir_valid(args.gt_dir)

    if args.out_dir is not None:
        common.check_dir_valid(args.out_dir)

    if not args.cpu and not torch.cuda.is_available():
        print("Warning, cannot use GPU as cuda is not available, will use CPU")
        args.cpu = True

    common.time_method(main, args)

