import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import fastai.vision as fvision
from radam import *

from PatchExtractor import PatchExtractor, DrawHelper
from ObjDetecPatchSamplerDataset import ObjDetecPatchSamplerDataset
import common
import entropy

class CustomModel(object):
    def __init__(self, model_path, output_dir, use_cpu):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda') if not use_cpu and torch.cuda.is_available() else torch.device('cpu')

    def load_model(self):
        raise NotImplementedError

    def predict_imgs(self, ims):
        raise NotImplementedError

    def prepare_img_for_inference(self, im):
        raise NotImplementedError

    def show_preds(self, img_arr, preds, title, fname):
        raise NotImplementedError


class ClassifModel(CustomModel):
    def __init__(self, model_path, output_dir, use_cpu, with_entropy):
        super().__init__(model_path, output_dir, use_cpu)
        fvision.defaults.device = torch.device(self.device)
        self.load_model()
        self.with_entropy = with_entropy
        if self.with_entropy:
            entropy.convert_learner(self.learner)

    def load_model(self):
        self.learner = fvision.load_learner(os.path.dirname(self.model_path), os.path.basename(self.model_path))

    def prepare_img_for_inference(self, ims):
        return [fvision.Image(fvision.pil2tensor(im, np.float32).div_(255)) for im in ims]

    def predict_imgs(self, ims):
        ims = self.prepare_img_for_inference(ims)
        preds = [[self.learner.predict(im)] for im in ims]
        if self.with_entropy:
            entropy.switch_custom_dropout(self.learner.model, True)
            entropy_preds = [self.learner.predict_with_mc_dropout(im) for im in ims]
            entropy.switch_custom_dropout(self.learner.model, False)
            preds = [p + pe for p, pe in zip(preds, entropy_preds)]
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

    def prepare_predictions(self, pm_preds, topk=3, d=2, entropy_thresh=1.25):
        trad = {'arme': 'Arm', 'beine': 'Leg', 'fusse': 'Feet', 'hande': 'Hand', 'kopf': 'Head', 'other': 'Other',
                'stamm': 'Trunk'}
        labels = [trad[c] for c in self.learner.data.classes]
        assert len(labels) >= topk, "Topk greater than the number of classes"
        colors = (['k', 'g', 'c', 'm', 'w', 'y', 'b', 'r'] * max(1, int(len(labels)/8)))[:len(labels)]

        # Each image has a list of preds. If no entropy this list contains a single element.
        pms, preds = zip(*pm_preds)
        # Create tensor for each image: shape is entropy sample x nclasses
        preds = [torch.cat([elem[2].unsqueeze(0) for elem in sublst], dim=0) for sublst in preds]
        # Create tensor for whole batch: shape is entropy sample x bs x nclasses
        preds = torch.cat([p.unsqueeze(1) for p in preds], dim=1)

        if self.with_entropy:
            entropy_ims = entropy.entropy(preds)
            topk_idx, topk_p, topk_std = entropy.custom_top_k_preds(preds, topk)
        else:
            topk_p, topk_idx = preds.squeeze(0).topk(topk, axis=1)

        res = {}
        for i, (pm, idxs, probs) in enumerate(zip(pms, topk_idx, topk_p)):
            res[pm['patch_path']] = [(f'{labels[idx]}: {p:.{d}f}', colors[idx]) for idx, p in zip(idxs, probs)]
            if self.with_entropy:
                res[pm['patch_path']] = [(v0 + f' \u00B1 {s:.{d}f}', v1)
                                         for (v0, v1), s in zip(res[pm['patch_path']], topk_std[i])]
                res[pm['patch_path']].append((f'Entropy: {entropy_ims[i]:.{d}f}',
                                              'b' if entropy_ims[i] < entropy_thresh else 'r'))
        return res

    def correct_predictions(self, preds, neighbors, cats, consider_top=2, method=2):
        return preds
        if method == 1:
            pred_probs = {k: v[2].numpy() for k, v in preds.items()}
            preds = {k: [f'orig: {str(v[0])}'] for k, v in preds.items()}
            summed_neighbors_preds = {k: sum([pred_probs[p] for p in neighbors[k]]) for k in preds.keys()}
            for patch in preds.keys():
                neighbors_preds = summed_neighbors_preds[patch]
                topk = neighbors_preds.argsort()[::-1][:consider_top]
                top_preds = list(zip(np.array(self.classes)[topk], neighbors_preds[topk]))
                preds[patch] = [f'{i + 1}: {p[0]}' for i, p in enumerate(top_preds)] + preds[patch]
            return preds
        elif method == 2:
            pred_probs = {k: v[2].numpy() for k, v in preds.items()}
            res = {k: [f'orig: {str(v[0])}'] for k, v in preds.items()}
            for k in res.keys():
                orig = int(preds[k][1])
                neigh_preds = {}
                for neighbor in neighbors[k]:
                    for top in pred_probs[neighbor].argsort()[::-1][:consider_top]:
                        neigh_preds[top] = neigh_preds.get(top, 0) + 1
                most_probable = [t[0] for t in sorted(neigh_preds.items(), key=lambda item: item[1])[::-1][:consider_top]]
                if orig not in most_probable:
                    res[k].append(f'corr: {cats[int(most_probable[0])]}')
            return res


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



class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im, target):
        for t in self.transforms:
            im, target = t(im, target)
        return im, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            # if "keypoints" in target:
            #     keypoints = target["keypoints"]
            #     keypoints = _flip_coco_person_keypoints(keypoints, width)
            #     target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = torchvision.transforms.functional.to_tensor(image)
        return image, target


class ObjDetecModel(CustomModel):
    def __init__(self, classes, model_path, output_dir, use_cpu):
        super().__init__(model_path, output_dir, use_cpu)
        self.classes = classes
        self.n_classes = len(self.classes)
        self.load_model()
        self.model.to(self.device)
        cmap = plt.get_cmap('tab20b')
        self.bbox_colors = random.sample([cmap(i) for i in np.linspace(0, 1, self.n_classes)], self.n_classes)

    def load_model(self):
        self.model = self.get_instance_segmentation_model()
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))

    def get_instance_segmentation_model(self):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.n_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, self.n_classes)
        return model

    def get_transforms(self):
        return Compose([ToTensor()])

    def prepare_img_for_inference(self, im):
        transform = self.get_transforms()
        img_tensor = transform(im, None)[0]
        return img_tensor.to(self.device)

    def predict_imgs(self, ims):
        self.model.eval()
        return self.preds_to_cpu(self.model([self.prepare_img_for_inference(im) for im in ims]))

    def preds_to_cpu(self, preds):
        # TODO: return masks as well
        # cpu_preds =
        # for pred in preds:
        #     cpu_preds.append()
        #     pred_class = [self.classes[i] for i in list(pred['labels'].cpu().numpy())]
        #     pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred['boxes'].cpu().detach().numpy())]
        #     pred_score = list(pred['scores'].cpu().detach().numpy())
        return [{k: v.cpu().detach().numpy() for k, v in pred.items()} for pred in preds]

    def filter_preds_by_conf(self, preds, min_conf):
        # memory error
        #return [{k: v[np.where(pred['scores'] >= min_conf)] for k, v in pred.items()} for pred in preds
        #        if pred['labels'].size > 0]
        idx = np.where(preds['scores'] >= min_conf)
        if preds['obj_ids'][idx].size == 0:
            return []
        remaining_objs = preds['obj_ids'][idx]
        preds['masks'] = {c: sum([self.get_obj_mask(obj_id, preds['masks'][c], sub_obj_id=False)
                            for obj_id in remaining_objs]) for c in preds['masks'].keys()}
        for k in preds.keys():
            if k != 'masks':
                preds[k] = preds[k][idx]
        return preds

    def show_preds(self, im, preds, title='Predictions', fname='predictions.jpg', show_bbox=False, transparency=True):
        plt.figure()
        fig, ax = plt.subplots(figsize=(20, 20))
        plt.axis('off')
        plt.title(title, fontsize=40, pad=10)
        ax.set_axis_off()
        ax = fig.add_subplot(1, 2, 1)
        ax.set_axis_off()
        ax.title.set_text('Original image')
        ax.title.set_fontsize(24)
        ax.imshow(im)
        ax = fig.add_subplot(1, 2, 2)
        ax.set_axis_off()
        ax.title.set_text('Predictions')
        ax.title.set_fontsize(24)
        img_arr = np.copy(im)
        if len(preds) == 0:
            cv2.putText(img_arr, 'NOTHING DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=3)
            ax.imshow(img_arr)
        else:
            preds = preds[0]
            classes = np.array(self.classes)
            pred_class = classes[preds['labels']]
            if transparency:
                alpha = .3
                for i, tup in enumerate(zip(preds['labels'], preds['obj_ids'])):
                    c, obj_id = tup
                    mask = self.get_obj_mask(obj_id, preds['masks'][c])[0][0]
                    thresh = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)
                    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1]
                    color = tuple(int(j * 255) for j in self.bbox_colors[int(np.where(classes == pred_class[i])[0])][:-1])
                    cv2.drawContours(img_arr, contours, 0, color, 2)
                mask = sum([self.get_obj_mask(obj_id, preds['masks'][c]) for obj_id in preds['obj_ids']])
                mask = np.dstack([cv2.threshold(mask[0][0], 0.5, 255, cv2.THRESH_BINARY)[1].astype(np.uint8)] * 3)
                img_arr = cv2.addWeighted(img_arr, 1 - alpha, mask, alpha, 0)
                #img_arr = img_arr * (1.0 - alpha) + mask * alpha
            else:
                for i, tup in enumerate(zip(preds['labels'], preds['obj_ids'])):
                    c, obj_id = tup
                    mask = self.get_obj_mask(obj_id, preds['masks'][c])[0][0]
                    color = tuple(int(i*255) for i in self.bbox_colors[int(np.where(classes == pred_class[i])[0])][:-1])
                    mask = mask
                    img_arr[mask > .8] = color
            ax.imshow(img_arr)
            box_w = preds['boxes'][:, 2] - preds['boxes'][:, 0]
            box_h = preds['boxes'][:, 3] - preds['boxes'][:, 1]
            printed = {k: False for k in classes}
            for i, boxes in enumerate(preds['boxes']):
                color = self.bbox_colors[int(np.where(classes == pred_class[i])[0])]
                bbox = plt_patches.Rectangle((boxes[0], boxes[1]), box_w[i], box_h[i], linewidth=2, edgecolor=color,
                                             facecolor='none')
                if show_bbox or not printed[pred_class[i]]:
                    ax.add_patch(bbox)
                    if not printed[pred_class[i]]:
                        printed[pred_class[i]] = True
                        plt.text(boxes[0], boxes[3], s=pred_class[i], color='white', verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
        if self.output_dir is not None:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=300)
        plt.show()
        plt.close()

    def adjust_bboxes_to_full_img(self, im, pm_preds, patch_size):
        preds = []
        for pm, pred in pm_preds:
            y, x = pm['idx_h'], pm['idx_w']
            empty_mask = np.zeros((pred['masks'].shape[0], 1, *im.shape[:-1]))
            empty_mask[:, :, y:y+patch_size, x:x+patch_size] = pred['masks']
            pred['masks'] = empty_mask
            pred['boxes'] += np.ones(pred['boxes'].shape) * np.array([x, y, x, y])
            preds.append(pred)
        return preds

    def merge_preds(self, preds):
        merged_preds = {k: [] for k in preds[0].keys()}
        for pred in preds:
            for k in merged_preds.keys():
                merged_preds[k].extend(list(pred[k]))

        merged_preds = self.merge_overlapping_dets(merged_preds)
        merged_preds['obj_ids'] = list(range(1, np.array(merged_preds['labels']).size+1))
        merged_mask = {k: np.zeros((1, 1, *merged_preds['masks'][0].shape[1:]), dtype=np.float64) for k in np.unique(merged_preds['labels'])}
        for i, obj_id in enumerate(merged_preds['obj_ids']):
            idx = np.where(merged_preds['masks'][i] > 0)
            merged_mask[merged_preds['labels'][i]][0][idx] = merged_preds['masks'][i][idx] + obj_id
        merged_preds['masks'] = merged_mask
        merged_preds = {k: np.array(v) if k != 'masks' else v for k, v in merged_preds.items()}
        return merged_preds

    def get_obj_mask(self, obj_id, merged_mask, sub_obj_id=True):
        idx = np.where(np.logical_and(0 <= (merged_mask - obj_id), (merged_mask - obj_id) < 1))
        mask = np.zeros((merged_mask.shape))
        if sub_obj_id:
            mask[idx] = merged_mask[idx] - obj_id
        else:
            mask[idx] = merged_mask[idx]
        return mask

    def merge_overlapping_dets(self, merged_preds):
        preds_overlap_merged = {k: [] for k in merged_preds.keys()}
        merged = []
        for i in range(len(merged_preds['labels'])):
            if i in merged:
                continue
            b1 = merged_preds['boxes'][i]
            overlap = [i]
            has_merged = True
            while has_merged:
                has_merged = False
                for j in range(len(merged_preds['labels'])):
                    if merged_preds['labels'][i] != merged_preds['labels'][j] or j in overlap:
                        continue
                    b2 = merged_preds['boxes'][j]
                    if ObjDetecModel.intersect(b1, b2):
                        b1 = [*[min(b1[i], b2[i]) for i in range(2)], *[max(b1[i+2], b2[i+2]) for i in range(2)]]
                        overlap.append(j)
                        has_merged = True
            preds_overlap_merged['labels'].append(merged_preds['labels'][i])
            preds_overlap_merged['boxes'].append(b1)
            preds_overlap_merged['scores'].append(np.mean([merged_preds['scores'][n] for n in overlap]))
            preds_overlap_merged['masks'].append(sum([merged_preds['masks'][n] for n in overlap]))
            # not 1 because otherwise, may confuse with next obj_id
            preds_overlap_merged['masks'][-1][preds_overlap_merged['masks'][-1] > 1] = .999
            merged.extend(overlap)
        return preds_overlap_merged

    def save_dets(self, img_id, preds, gt=None):
        if len(preds) == 0:
            gt_pred = ((gt['labels'], gt['boxes']), ([], [], []))
        else:
            gt_pred = ((gt['labels'], gt['boxes']), tuple(preds[k] for k in ['labels', 'scores', 'boxes']))
        self.write_gt_pred(str(img_id)+'.txt', gt_pred)

    def write_gt_pred(self, filename, gt_pred):
        gt_dir = os.path.join(self.output_dir, 'groundtruths')
        det_dir = os.path.join(self.output_dir, 'detections')
        if not os.path.exists(gt_dir):
            os.makedirs(gt_dir)
            os.makedirs(det_dir)
        gts, preds = gt_pred
        with open(os.path.join(gt_dir, filename), "w") as file:
            for label, bbox in zip(gts[0], gts[1]):
                file.write(f'{self.classes[int(label)]} {" ".join(map(str, bbox))}\n')
        with open(os.path.join(det_dir, filename), "w") as file:
            if len(preds[0]) > 0:
                for label, score, bbox in zip(preds[0], preds[1], preds[2]):
                    file.write(f'{self.classes[int(label)]} {score} {" ".join(map(str, bbox))}\n')

    @staticmethod
    def intersect(box1, box2):
        xmin, ymin, xmax, ymax = box1
        xmin2, ymin2, xmax2, ymax2 = box2
        return not (xmin > xmax2 or xmin2 > xmax or ymin > ymax2 or ymin2 > ymax)

    @staticmethod
    def extract_mask_objs(mask):
        objs = ObjDetecPatchSamplerDataset.extract_mask_objs(mask)
        if objs is None: return None
        else: return objs['obj_count'], objs['boxes'], objs['obj_masks']

    @staticmethod
    def get_img_gt(img_path):
        file, ext = os.path.splitext(os.path.basename(img_path))
        img_dir = os.path.dirname(img_path)
        root_dir = os.path.dirname(img_dir)
        masks = [cv2.imread(os.path.join(mask_path, file + '.png'), cv2.IMREAD_UNCHANGED) for mask_path
                 in [os.path.join(root_dir, m) for m in sorted(os.listdir(root_dir)) if m.startswith('masks_')]]
        objs = list(zip(*filter(None.__ne__, map(ObjDetecModel.extract_mask_objs, masks))))
        if not objs:  # check list empty
            return None
        objs[0] = tuple((i + 1) * np.ones(n_obj, dtype=np.int) for i, n_obj in enumerate(objs[0]))
        classes, boxes, masks = (np.concatenate(val) for val in objs)
        return {'labels': classes, 'boxes': boxes, 'masks': masks.reshape((masks.shape[0], 1, *masks.shape[1:])).astype(np.float)}


def main(args):
    img_list = [args.img] if args.img is not None else common.list_files(args.img_dir, full_path=True)

    if args.obj_detec:
        classes = ['__background__', 'pustule', 'brown_spot'] if args.classes is None else args.classes
        model = ObjDetecModel(classes, args.model, args.out_dir, args.cpu)
    else:
        if args.effnet:
            model = EffnetClassifModel(args.model, args.out_dir, args.cpu, args.entropy, args.data_sample,
                                       args.bs, args.input_size)
        else:
            model = ClassifModel(args.model, args.out_dir, args.cpu, args.entropy)

    patcher = PatchExtractor(args.ps)
    for img_id, img_path in enumerate(img_list):
        print(f"Image {img_id}/{len(img_list)}: {img_path}")
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = cv2.cvtColor(patcher.load_image(img_path), cv2.COLOR_BGR2RGB)

        if args.obj_detec:
            gt = ObjDetecModel.get_img_gt(img_path)
            if args.out_dir is not None and gt is None:
                raise Exception("No ground truth available, but user requested to save dets")
            if args.show_gt:
                model.show_preds(im, [gt], title=f'Ground Truth for {file}{ext}', fname=f'{file}_00_gt{ext}')

        patch_maps = patcher.patch_grid(img_path, im)
        b_pms = common.batch_list(patch_maps, args.bs)

        pm_preds = []
        for batch in tqdm(b_pms) if len(b_pms) > 1 else b_pms:
            patches = [PatchExtractor.extract_patch(im, pm['ps'], pm['idx_h'], pm['idx_w']) for pm in batch]
            batch_preds = model.predict_imgs(patches)
            pm_preds.extend(zip(batch, batch_preds))
            # TODO handle args.heatmap

        if args.obj_detec:
            pm_preds = [p for p in pm_preds if p[1]['labels'].size > 0]
            if len(pm_preds) > 0:
                preds = model.adjust_bboxes_to_full_img(im, pm_preds, args.ps)
                preds = [model.merge_preds(preds)]
                if args.out_dir is not None:
                    model.save_dets(img_id, preds[0], gt=gt)
                preds = [model.filter_preds_by_conf(preds[0], args.conf_thresh)]
            else:
                preds = []
            title = f'Predictions with confidence > {args.conf_thresh}'
            plot_name = f'{file}_01_conf_{args.conf_thresh}{ext}'
        else:
            preds = model.prepare_predictions(pm_preds)
            neighbors = {p['patch_path']: patcher.neighboring_patches(p, im.shape, d=1) for p in patch_maps}
            # preds = model.correct_predictions(preds, neighbors, cats=model.learner.data.classes)
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
    parser.add_argument('--classif', action='store_true', help="Applies classification model")
    parser.add_argument('--data-sample', type=str, help="optional, needed to recreate effnet learner")
    parser.add_argument('--effnet', action='store_true', help="efficientnet model if set, need --data-sample")
    parser.add_argument('--heatmap', action='store_true', help="For classif, creates gradcam image")
    parser.add_argument('--entropy', action='store_true', help="Compute image entropy")

    # obj detec args
    parser.add_argument('--obj-detec', action='store_true', help="Applies object detection model")
    parser.add_argument('--conf-thresh', default=.5, type=float, help="Confidence threshold for detections")
    parser.add_argument('--classes', type=str, nargs='*', help="classes for detections")
    parser.add_argument('--show-gt', action='store_true', help="Show gt of object detection dataset")
    args = parser.parse_args()

    if args.img is not None:
        common.check_file_valid(args.img)
    else:
        common.check_dir_valid(args.img_dir)

    if args.out_dir is not None:
        common.check_dir_valid(args.out_dir)

    if args.obj_detec and args.classif:
        raise Exception("Error, both object detection and classif are set")
    elif not args.obj_detec and not args.classif:
        raise Exception("Error, neither object detection or classif were set")

    if args.effnet:
        assert args.data_sample is not None, "Efficient net model needs a sample data dir to be loaded"

    if not args.cpu:
        common.maybe_set_gpu(args.gpuid, args.num_gpus)

    common.time_method(main, args)

