import os
import cv2
import types
import numpy as np
from tqdm import tqdm
from functools import partial
import matplotlib.pyplot as plt

import fastai.vision as fvision
from radam import *

from PatchExtractor import PatchExtractor, DrawHelper
import common
import entropy


class ClassifModel:
    def __init__(self, model_path, output_dir, use_cpu, with_entropy, n_times=10):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda') if not use_cpu and torch.cuda.is_available() else torch.device('cpu')
        fvision.defaults.device = torch.device(self.device)
        self.load_model()
        self.with_entropy = with_entropy
        if self.with_entropy:
            entropy.convert_learner(self.learner)
            self.n_times = n_times

    def load_model(self):
        self.learner = fvision.load_learner(os.path.dirname(self.model_path), os.path.basename(self.model_path))

    def prepare_img_for_inference(self, ims):
        ims = [fvision.Image(fvision.pil2tensor(im, np.float32).div_(255)) for im in ims]
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

    def gradcam(self, im, patches, patch_size=512):
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
        return im

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
        res.corrected = np.zeros(res.patches.size, dtype=np.bool)
        return res


def correct_predictions(model_preds, patch_size, neigh_dist=1, topk=2, with_entropy=False,
                        entropy_thresh=1.25, method="in_top_neigh_probs"):
    """Corrects individual patch predictions by looking at neighboring patches"""
    _, neigh_pidx, _, pidx_groups = PatchExtractor.get_neighbors_dict(model_preds.patches, neigh_dist, patch_size)
    # group is the number of neighbors, if patch has n neighbors it belongs to group n

    # indexes of patches to be corrected
    if with_entropy:
        corr_all = (model_preds.entropy > entropy_thresh).nonzero()[0]
        model_preds.orig_std = model_preds.std.detach().clone()
        # TODO: update neigh_pidx, pidx_groups so that neighbors with high entropy are removed
    else:
        corr_all = np.arange(model_preds.patches.size)

    corr_groups = pidx_groups[corr_all]     # groups of patches to be corrected
    idx_sort = np.argsort(corr_groups)      # idx to sort corr_groups so that the same groups are adjacent
    corr_groups, idx_start = np.unique(corr_groups[idx_sort], return_index=True)
    corr_idx = np.split(idx_sort, idx_start[1:])    # lst of arr of idx of patches of same group
    if corr_groups[0] == 0:     # if no neighbors, cannot correct patch label
        corr_groups, corr_idx = corr_groups[1:], corr_idx[1:]

    model_preds.orig_preds = model_preds.preds.detach().clone()
    model_preds.orig_pred_class = model_preds.pred_class.detach().clone()

    for idx in corr_idx:    # iterate over all groups of patches with same number of neighbors
        cidx = corr_all[idx]
        # neigh_pidx orig shape irregular => orig type is object => need to cast to ints in order to use as index
        cidx_neighs = np.array(neigh_pidx[cidx].tolist())
        if method == "mean_neigh_probs":
            # interp.preds[cidx] shape is nb_images x nb_classes
            # interp.preds[idx_neighs] shape is nb_images x nb_neighbors x nb_classes
            model_preds.preds[cidx] = model_preds.preds[cidx_neighs,].mean(axis=1)
            model_preds.pred_class[cidx] = model_preds.preds[cidx].argmax(axis=1)
            # interp.std[idx_neighs] shape is nb_images x nb_neighbors x nb_classes
            if with_entropy:
                model_preds.std[cidx] = model_preds.std[cidx_neighs,].mean(axis=1)
        elif method == "in_top_neigh_probs":
            topk_p, topk_idx = model_preds.preds[cidx_neighs,].topk(topk, axis=2)
            topk_p, topk_idx = map(partial(torch.flatten, start_dim=1), (topk_p, topk_idx))
            top = np.apply_along_axis(common.most_common, axis=1, arr=topk_idx.numpy(), top=topk, return_index=False)
            top = np.hstack([top, model_preds.pred_class[cidx].unsqueeze(1).numpy()])
            model_preds.pred_class[cidx] = torch.tensor(np.apply_along_axis(pred_among_most_probable, axis=1, arr=top))
    model_preds.corrected = model_preds.orig_pred_class != model_preds.pred_class

def pred_among_most_probable(arr):
    """Arr contains the most common neighbors preds + the model prediction as the last element"""
    if arr[-1] in arr[:-1]:
        return arr[-1]
    else:
        return arr[0]


class EffnetClassifModel(ClassifModel):
    def __init__(self, model_path, output_dir, use_cpu, with_entropy, data_sample, bs, input_size):
        test = 'strong_labels_test'
        train = 'strong_labels_train'
        self.data = fvision.ImageDataBunch.from_folder(path=data_sample, train=train, valid=test, size=input_size, bs=bs)
        self.data.normalize(fvision.imagenet_stats)
        self.classes = self.data.classes
        self.n_classes = len(self.classes)
        super().__init__(model_path, output_dir, use_cpu, with_entropy)

    def load_model(self):
        from efficientnet_pytorch import EfficientNet
        model = EfficientNet.from_name('efficientnet-b6')
        model._fc = torch.nn.Linear(model._fc.in_features, self.data.c)
        common.load_custom_pretrained_weights(model, self.model_path)
        self.learner = fvision.Learner(model=model, data=self.data)


def main(args):
    img_list = [args.img] if args.img is not None else common.list_files(args.img_dir, full_path=True)

    if args.effnet:
        model = EffnetClassifModel(args.model, args.out_dir, args.cpu, args.with_entropy, args.data_sample,
                                   args.bs, args.input_size)
    else:
        model = ClassifModel(args.model, args.out_dir, args.cpu, args.with_entropy)

    patcher = PatchExtractor(args.ps)
    for img_id, img_path in enumerate(img_list):
        print(f"Image {img_id}/{len(img_list)}: {img_path}")
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = cv2.cvtColor(patcher.load_image(img_path), cv2.COLOR_BGR2RGB)

        patch_maps = patcher.patch_grid(img_path, im)
        b_pms = common.batch_list(patch_maps, args.bs)

        preds = []
        for batch in tqdm(b_pms) if len(b_pms) > 1 else b_pms:
            patches = [PatchExtractor.extract_patch(im, pm['ps'], pm['idx_h'], pm['idx_w']) for pm in batch]
            batch_preds = model.predict_imgs(patches)
            preds.append(batch_preds)
            # TODO handle args.heatmap

        preds = model.prepare_predictions(patch_maps, preds)
        preds.classes = model.learner.data.classes
        if args.pred_correction:
            correct_predictions(preds, args.ps, with_entropy=args.with_entropy, entropy_thresh=args.entropy_thresh,
                                method=args.corr_method)
        title = f'Body localization'
        plot_name = f'{file}_body_loc{ext}'

        if args.draw_patches:
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
    parser.add_argument('--input-size', default=256, type=int, help="model input size")
    parser.add_argument('--draw-patches', action='store_true', help="Draws patches")
    parser.add_argument('--out-dir', type=str, help="if save-out set, output dir absolute path")
    parser.add_argument('--no-graphs', action='store_true', help="Do not create graphs")
    parser.add_argument('--cpu', action='store_true', help="Use CPU (defaults use cuda if available)")
    parser.add_argument('--gpuid', type=int, help="For single gpu, gpu id to be used")
    common.add_multi_gpus_args(parser)

    # classif args
    parser.add_argument('--data-sample', type=str, help="optional, needed to recreate effnet learner")
    parser.add_argument('--effnet', action='store_true', help="efficientnet model if set, need --data-sample")
    parser.add_argument('--heatmap', action='store_true', help="For classif, creates gradcam image")
    parser.add_argument('--with-entropy', action='store_true', help="Compute image entropy")
    parser.add_argument('--entropy-thresh', default=1.25, type=float, help="Threshold to find patch to correct")
    parser.add_argument('--pred-correction', action='store_true', help="Corrects preds by checking neighboring patches")
    parser.add_argument('--corr-method', type=str, default='in_top_neigh_probs', help="in_top_neigh_probs or mean_neigh_probs")

    args = parser.parse_args()

    if args.img is not None:
        common.check_file_valid(args.img)
    else:
        common.check_dir_valid(args.img_dir)

    if args.out_dir is not None:
        common.check_dir_valid(args.out_dir)

    if args.effnet:
        assert args.data_sample is not None, "Efficient net model needs a sample data dir to be loaded"

    if not args.cpu:
        common.maybe_set_gpu(args.gpuid, args.num_gpus)

    if args.pred_correction:
        corr_methods = ['in_top_neigh_probs', 'mean_neigh_probs']
        assert args.corr_method in corr_methods, f"Correction method not supported, please select among {corr_methods}"

    common.time_method(main, args)

