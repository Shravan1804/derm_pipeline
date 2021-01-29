import os
import types
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

from radam import *

import common
import entropy
from infer_images import FastaiModel
from PatchExtractor import PatchExtractor, DrawHelper


class ClassifModel(FastaiModel):
    def __init__(self, model_path, output_dir, use_cpu, with_entropy, bs, n_times=10):
        super().__init__(model_path, use_cpu, bs)
        self.output_dir = output_dir
        self.with_entropy = with_entropy
        if self.with_entropy:
            entropy.convert_learner(self.learner)
            self.n_times = n_times

    def prepare_img_for_inference(self, ims):
        ims = [fv.Image(fv.pil2tensor(im, np.float32).div_(255)) for im in ims]
        return torch.cat([self.learner.data.one_item(im)[0] for im in ims], dim=0)

    def predict_imgs(self, ims):
        """Returns tensor of shape n_times x n_images x n_classes"""
        ims = self.prepare_img_for_inference(ims)
        preds = self.learner.pred_batch(batch=[ims, -1], with_dropout=False).unsqueeze(0)
        if self.with_entropy:
            entropy.switch_custom_dropout(self.learner.model, True)
            entropy_preds = [self.learner.pred_batch(batch=[ims, -1], with_dropout=True) for _ in range(self.n_times)]
            entropy.switch_custom_dropout(self.learner.model, False)
            entropy_preds = torch.cat([e.unsqueeze(0) for e in entropy_preds], dim=0)
            preds = torch.cat([preds, entropy_preds], dim=0)
        return preds

    """def gradcam(self, im, patches, patch_size=512):
        from cnn_viz.visualisation.core import GradCam
        from cnn_viz.visualisation.core.utils import image_net_postprocessing
        from cnn_viz.utils import tensor2img
        vis = GradCam(self.learner.model, torch.device('cpu'))
        print("Computing heatmap")
        for pm, p in tqdm(patches):
            xb, _ = self.learner.data.one_item(self.prepare_img_for_inference([p]), detach=False, denorm=False)
            hm = vis(xb, self.learner.model._conv_head, postprocessing=image_net_postprocessing)[0]
            hm = tensor2img(fvision.denormalize(hm, *[torch.FloatTensor(x) for x in fvision.imagenet_stats]) * 255)
            hm = cv2.resize(hm.astype(np.uint8), (patch_size, patch_size))
            h, w = pm['idx_h'], pm['idx_w']
            im[h:h + patch_size, w:w + patch_size] = hm
        return im"""

    def show_preds(self, img_arr, preds, title, fname):
        preds = self.predictions_to_labels(preds)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(20, 20))
        for patch, pred in preds.items():
            h, w = PatchExtractor.get_position(patch)
            for i, (p, c) in enumerate(pred):
                if i > 0 and not (self.with_entropy and i == len(pred)-1):  # entropy value keeps its own color
                    c = 'y'
                plt.text(50 + w, (i+1)*50 + h, p, color=c, fontsize=8, fontweight='bold')
        ax.imshow(img_arr)
        plt.axis('off')
        plt.title(title, fontsize=42)
        if self.output_dir is not None:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=300)
        plt.show()
        plt.close()

    def predictions_to_labels(self, model_preds, topk=3, deci=2, entropy_thresh=1.25):
        trad = {'arme': 'Arm', 'beine': 'Leg', 'fusse': 'Feet', 'hande': 'Hand', 'kopf': 'Head', 'other': 'Other',
                'stamm': 'Trunk'}
        labels = [trad[c] for c in self.learner.data.classes]
        assert len(labels) >= topk, "Topk greater than the number of classes"
        colors = (['k', 'g', 'c', 'm', 'w', 'lime', 'maroon', 'darkorange'] * max(1, int(len(labels)/8)))[:len(labels)]
        #colors = ['y'] * len(labels)

        topk_p, topk_idx = model_preds.preds.topk(topk, axis=1)
        if self.with_entropy:
            topk_std = model_preds.std.gather(dim=1, index=topk_idx)

        res = {}
        for i, (pm, idxs, probs) in enumerate(zip(model_preds.patches, topk_idx, topk_p)):
            res[pm] = [(f'{labels[idx]}: {p:.{deci}f}', colors[idx]) for idx, p in zip(idxs, probs)]
            res[pm] = [(f'{"(corr) " if model_preds.corrected[i] else ""}{r}', c) for r, c in res[pm]]
            if self.with_entropy:
                res[pm] = [(v0 + f' \u00B1 {s:.{deci}f}', v1) for (v0, v1), s in zip(res[pm], topk_std[i])]
                entropy_color = 'b' if model_preds.entropy[i] < entropy_thresh else 'r'
                res[pm].append((f'Entropy: {model_preds.entropy[i]:.{deci}f}', entropy_color))
        return res

    def prepare_predictions(self, pms, preds):
        res = types.SimpleNamespace()
        res.patches = np.array([pm['patch_path'] for pm in pms])
        # cats the list of batches, produces tensor of shape n_times x n_images x n_classes
        res.entropy_preds = torch.cat(preds, dim=1)
        if self.with_entropy:
            res.entropy = entropy.entropy(res.entropy_preds)
            res.std = res.entropy_preds.std(axis=0)
            res.preds = res.entropy_preds.mean(axis=0)
        else:
            res.preds = res.entropy_preds.squeeze(0).detach().clone()
        res.pred_class = res.preds.argmax(axis=1)
        res.corrected = torch.zeros_like(res.pred_class, dtype=bool)
        return res


def correction_candidates(model_preds, neigh_pidx, with_entropy):
    if with_entropy:
        # all patch with entropy higher than most can be candidates
        candidates = torch.tensor(model_preds.entropy > np.quantile(model_preds.entropy, [.85])[0])
    else:
        candidates = torch.zeros_like(model_preds.pred_class, dtype=bool)

    top2_p, top2_idx = model_preds.preds.topk(2, axis=1)

    for pid, (pred, top2_i, top2_p) in enumerate(zip(model_preds.pred_class, top2_idx, top2_p)):
        # patch with "other" class in top 2 preds must be corrected since model is so confident with this class
        candidates[pid] = candidates[pid] or (5 in top2_i[1:])
        # unique prediction among neighbors
        candidates[pid] = candidates[pid] or (pred not in model_preds.pred_class[neigh_pidx[pid]])
        # Top pred proba should be at least 15% larger than second top pred proba
        candidates[pid] = candidates[pid] or (top2_p[0] - top2_p[1] < .15)

    # model is almost never wrong with the "other" class
    candidates = candidates & ~(model_preds.pred_class == 5)

    return candidates.numpy()


def correct_patch_with_other_class(model_preds, corr_all, with_entropy=False):
    """Sets the preds to other if other present in top 2 preds, returns unchanged patch indexes
    Changes the preds, assumes already backed up"""
    top2_p, top2_idx = model_preds.preds[corr_all, ].topk(2, axis=1)
    change = np.apply_along_axis(lambda top2_i: 5 in top2_i[1:], axis=1, arr=top2_idx.numpy())
    idx = corr_all[change]
    model_preds.pred_class[idx] = 5
    model_preds.preds[idx, model_preds.pred_class[idx]] = 1
    if with_entropy:
        model_preds.std[idx, model_preds.pred_class[idx]] = 0
    return corr_all[~change]


def correct_predictions(model_preds, neigh_dist=1, topk=2, with_entropy=False, method="mean_neigh_probs"):
    """Corrects individual patch predictions by looking at neighboring patches"""
    _, neigh_pidx, _, pidx_groups = PatchExtractor.get_neighbors_dict(model_preds.patches, neigh_dist)
    # group is the number of neighbors, if patch has n neighbors it belongs to group n

    # indexes of patches to be corrected, numpy nonzero returns tuple while torch returns tensor directly
    corr_all = correction_candidates(model_preds, neigh_pidx, with_entropy).nonzero()[0]
    if corr_all.size == 0: return

    # backup preds before correction
    model_preds.orig_preds = model_preds.preds.detach().clone()
    model_preds.orig_pred_class = model_preds.pred_class.detach().clone()
    if with_entropy:
        model_preds.orig_std = model_preds.std.detach().clone()

    corr_all = correct_patch_with_other_class(model_preds, corr_all)
    if corr_all.size == 0: return

    corr_groups = pidx_groups[corr_all]     # groups of patches to be corrected
    idx_sort = np.argsort(corr_groups)      # idx to sort corr_groups so that the same groups are adjacent
    corr_groups, idx_start = np.unique(corr_groups[idx_sort], return_index=True)
    corr_idx = np.split(idx_sort, idx_start[1:])    # lst of arr of idx of patches of same group
    if corr_groups.size > 0 and corr_groups[0] == 0:     # if no neighbors, cannot correct patch label
        corr_groups, corr_idx = corr_groups[1:], corr_idx[1:]

    for idx in corr_idx:    # iterate over all groups of patches with same number of neighbors
        cidx = corr_all[idx]
        # neigh_pidx orig shape irregular => orig type is object => need to cast to ints in order to use as index
        cidx_neighs = np.array(neigh_pidx[cidx].tolist())
        # METHODS MUST UPDATE model_preds.preds AND model_preds.std, AS IT IS NEEDED IN predictions_to_labels
        if method == "mean_neigh_probs":
            # model_preds.preds[cidx] shape is nb_images x nb_classes
            # model_preds.preds[idx_neighs] shape is nb_images x nb_neighbors x nb_classes
            model_preds.preds[cidx] = model_preds.preds[cidx_neighs,].mean(axis=1)
            model_preds.pred_class[cidx] = model_preds.preds[cidx].argmax(axis=1)
            # model_preds.std[idx_neighs] shape is nb_images x nb_neighbors x nb_classes
            if with_entropy:
                model_preds.std[cidx] = model_preds.std[cidx_neighs,].mean(axis=1)
        elif method == "in_top_neigh_probs":
            # topk_idx, topk_p shapes are nb_images x nb_neighbors x topk
            topk_p, topk_idx = model_preds.preds[cidx_neighs,].topk(topk, axis=2)
            topk_p, topk_idx = map(partial(torch.flatten, start_dim=1), (topk_p, topk_idx))
            top = np.apply_along_axis(common.most_common, axis=1, arr=topk_idx.numpy(), top=topk, return_index=False)
            top = np.hstack([top, model_preds.pred_class[cidx].unsqueeze(1).numpy()])
            model_preds.pred_class[cidx] = torch.tensor(np.apply_along_axis(pred_among_most_probable, axis=1, arr=top))
            # update only the preds which were corrected
            change = cidx[model_preds.orig_pred_class[cidx] != model_preds.pred_class[cidx]]
            model_preds.preds[change, model_preds.pred_class[change]] = 1
            if with_entropy:
                model_preds.std[change, model_preds.pred_class[change]] = 0
    model_preds.corrected = model_preds.orig_pred_class != model_preds.pred_class


def pred_among_most_probable(arr):
    """Arr contains the most common neighbors preds + the model prediction as the last element"""
    return arr[-1] if arr[-1] in arr[:-1] else arr[0]


def main(args):
    img_list = [args.img] if args.img is not None else common.list_files(args.img_dir, full_path=True)

    model = ClassifModel(args.model, args.out_dir, args.cpu, args.with_entropy, args.bs)

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
            # TODO handle args.heatmap

        preds = model.prepare_predictions(patch_maps, preds)
        preds.classes = model.learner.data.classes
        if args.pred_correction:
            correct_predictions(preds, with_entropy=args.with_entropy, method=args.corr_method)
        title = f'Body localization'
        plot_name = f'{file}_body_loc.jpg'

        if not args.not_draw_patches:
            DrawHelper().draw_patches(im, patch_maps, args.ps)
        if not args.no_graphs:
            model.show_preds(im, preds, title=f'{title} for {file}{ext}', fname=plot_name)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Apply model to images")
    parser.add_argument('--img-dir', type=str, help="source image root directory absolute path")
    parser.add_argument('--img', type=str, help="absolute path, single image use, will take precedence over img_dir")
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('--bs', default=3, type=int, help="batch size")
    parser.add_argument('--ps', default=512, type=int, help="patch size")
    parser.add_argument('--not-draw-patches', action='store_true', help="Do not draws patches")
    parser.add_argument('--out-dir', type=str, help="if save-out set, output dir absolute path")
    parser.add_argument('--no-graphs', action='store_true', help="Do not create graphs")
    parser.add_argument('--cpu', action='store_true', help="Use CPU (defaults use cuda if available)")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    common.add_multi_gpus_args(parser)

    # classif args
    parser.add_argument('--heatmap', action='store_true', help="For classif, creates gradcam image")
    parser.add_argument('--with-entropy', action='store_true', help="Compute image entropy")
    parser.add_argument('--pred-correction', action='store_true', help="Corrects preds by checking neighboring patches")
    parser.add_argument('--corr-method', type=str, default='mean_neigh_probs', help="in_top_neigh_probs or mean_neigh_probs")

    args = parser.parse_args()

    if args.img is not None:
        common.check_file_valid(args.img)
    else:
        common.check_dir_valid(args.img_dir)

    if args.out_dir is not None:
        common.check_dir_valid(args.out_dir)

    if not args.cpu:
        common.maybe_set_gpu(args.gpuid, args.num_gpus)

    if args.pred_correction:
        corr_methods = ['in_top_neigh_probs', 'mean_neigh_probs']
        assert args.corr_method in corr_methods, f"Correction method not supported, please select among {corr_methods}"

    if not args.cpu and not torch.cuda.is_available():
        print("Warning, cannot use GPU as cuda is not available, will use CPU")
        args.cpu = True

    common.time_method(main, args)
