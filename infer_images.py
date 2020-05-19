import os
import cv2
import sys
import random
import argparse
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

class CustomModel(object):
    def __init__(self, seed, classes, model_path, output_dir):
        self.seed = seed
        self.classes = classes
        self.n_classes = len(self.classes)
        self.load_model(model_path)
        self.device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda')
        self.output_dir = output_dir

    def load_model(self, model_path):
        raise NotImplementedError

    def predict_imgs(self, ims):
        raise NotImplementedError

    def prepare_img_for_inference(self, im):
        raise NotImplementedError

    def show_preds(self, img_arr, preds, title, fname):
        raise NotImplementedError


class ClassifModel(CustomModel):
    def __init__(self, seed, model_path, output_dir):
        super().__init__(seed, [], model_path, output_dir)
        self.classes = self.model.data.classes
        self.n_classes = len(self.classes)
        fvision.defaults.device = torch.device(self.device)

    def load_model(self, model_path):
        self.model = fvision.load_learner(os.path.dirname(model_path), os.path.basename(model_path))

    def prepare_img_for_inference(self, im):
        t = fvision.pil2tensor(im, dtype=im.dtype)  # converts to numpy tensor
        t = fvision.Image(t.float() / 255.)  # Convert to float
        return t

    def predict_imgs(self, ims):
        return [self.model.predict(self.prepare_img_for_inference(im)) for im in ims]

    def show_preds(self, img_arr, preds, title, fname):
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(20, 20))
        colors = [(255, 255, 0), (255, 255, 0), (0, 0, 255)]
        for patch, pred in preds.items():
            h, w = PatchExtractor.get_position(patch)
            for i, p in enumerate(pred):
                cv2.putText(img_arr, p, (50 + w, (i+1)*50 + h), cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], thickness=3)
        ax.imshow(img_arr)
        plt.axis('off')
        plt.title(title, fontsize=42)
        if self.output_dir is not None:
            plt.savefig(os.path.join(self.output_dir, fname), dpi=300)
        plt.show()
        plt.close()

    def prediction_validation(self, preds, neighbors, consider_top=2):
        pred_probs = {k: v[2].numpy() for k, v in preds.items()}
        preds = {k: [f'orig: {str(v[0])}'] for k, v in preds.items()}
        summed_neighbors_preds = {k: sum([pred_probs[p] for p in neighbors[k]]) for k in preds.keys()}
        for patch in preds.keys():
            neighbors_preds = summed_neighbors_preds[patch]
            topk = neighbors_preds.argsort()[::-1][:consider_top]
            top_preds = list(zip(np.array(self.classes)[topk], neighbors_preds[topk]))
            preds[patch] = [f'{i+1}: {p[0]}' for i, p in enumerate(top_preds)] + preds[patch]
        return preds


class ObjDetecModel(CustomModel):
    def __init__(self, seed, classes, model_path, output_dir):
        super().__init__(seed, classes, model_path, output_dir)
        self.model.to(self.device)
        cmap = plt.get_cmap('tab20b')
        random.seed(self.seed)
        self.bbox_colors = random.sample([cmap(i) for i in np.linspace(0, 1, self.n_classes)], self.n_classes)

    def load_model(self, model_path):
        self.model = self.get_instance_segmentation_model()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

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

def plot_img(img_arr, title, output_path):
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_arr)
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

def unbatch(pm_preds):
    pms, preds = zip(*pm_preds)
    pms = [p for pm in pms for p in pm]
    preds = [p for pred in preds for p in pred]
    return zip(pms, preds)

def main():
    parser = argparse.ArgumentParser(description="Apply model to image patches")
    parser.add_argument('--img-dir', type=str, help="source image root directory absolute path")
    parser.add_argument('--img', type=str, help="absolute path, single image use, will take precedence over img_dir")
    parser.add_argument('--model', type=str, help="model path")
    parser.add_argument('-b', '--batch-size', default=3, type=int, help="batch size")
    parser.add_argument('-p', '--patch-size', default=512, type=int, help="patch size")
    parser.add_argument('-conf', '--conf-thresh', default=.5, type=float, help="Confidence threshold for detections")
    parser.add_argument('--draw-patches', action='store_true', help="Draws patches")
    parser.add_argument('--obj-detec', action='store_true', help="Applies object detection model")
    parser.add_argument('--classif', action='store_true', help="Applies classification model")
    parser.add_argument('--classes', type=str, nargs='*', help="classes for detections")
    parser.add_argument('--out-dir', type=str, help="if save-output set, output dir absolute path")
    parser.add_argument('--save-output', action='store_true', help="Save graphs")
    parser.add_argument('--save-dets', action='store_true', help="Save detections")
    parser.add_argument('--no-graphs', action='store_true', help="Do not create graphs")
    parser.add_argument('--show-gt', action='store_true', help="Show gt of object detection dataset")
    parser.add_argument('--seed', default=42, type=int, help="batch size")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.img is not None:
        if not os.path.exists(args.img):
            raise Exception("Error, specified image path doesn't exist:", args.img)
        img_list = [args.img]
    else:
        if args.img_dir is None or not os.path.exists(args.img_dir):
            raise Exception("Error, both --img and --img-dir are invalid")
        img_list = [os.path.join(args.img_dir, img) for img in sorted(os.listdir(args.img_dir))]

    if args.save_output or args.save_dets:
        if args.out_dir is None:
            raise Exception("Error, --save-output (or --save-dets) is set but no --out-dir")
        elif not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    else:
        args.out_dir = None

    if args.obj_detec and args.classif:
        raise Exception("Error, both object detection and classification are set")
    elif args.obj_detec:
        classes = ['__background__', 'pustule', 'brown_spot'] if args.classes is None else args.classes
        model = ObjDetecModel(args.seed, classes, args.model, args.out_dir)
    elif args.classif:
        model = ClassifModel(args.seed, args.model, args.out_dir)
    else:
        raise Exception("Error, neither object detection nor classification was chosen")
    patcher = PatchExtractor(args.patch_size)
    # pics problem causing pics
    #img_list = [os.path.join(args.img_dir, i) for i in ['run12_00012.jpg', 'run12_00023.jpg', 'run12_00028.jpg', 'run13_00014.jpg', 'run13_00015.jpg', 'run13_00016.jpg', 'run13_00097.jpg', 'run13_00114.jpg']]
    # test set pics
    # img_list = [os.path.join(args.img_dir, i) for i in ['run13_00099.jpg', 'run13_00016.jpg', 'run13_00083.jpg', 'run13_00014.jpg', 'run13_00123.jpg', 'run13_00073.jpg', 'run13_00143.jpg', 'run13_00064.jpg', 'run13_00117.jpg', 'run13_00150.jpg', 'run13_00002.jpg', 'run13_00110.jpg', 'run13_00028.jpg', 'run13_00032.jpg', 'run13_00074.jpg']]
    for img_id, img_path in enumerate(img_list):
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = cv2.cvtColor(patcher.load_image(img_path), cv2.COLOR_BGR2RGB)
        gt = ObjDetecModel.get_img_gt(img_path)
        if args.obj_detec and args.show_gt and not args.no_graphs:
            model.show_preds(im, [gt], title=f'Ground Truth for {file}{ext}', fname=f'{file}_00_gt{ext}')
        print("Creating patches for", img_path)
        pm = patcher.patch_grid(img_path, im)
        # create batches
        b_pm = [pm[i:min(len(pm), i+args.batch_size)] for i in range(0, len(pm), args.batch_size)]
        # lst of batches with a batch being a tuple containing lst of patch maps and lst of corresponding image patches
        b_patches = [(b, [patcher.extract_patch(im, pms['idx_h'], pms['idx_w']) for pms in b]) for b in b_pm]
        print("Applying model to patches")
        pm_preds = unbatch([(pms, model.predict_imgs(ims)) for pms, ims in tqdm(b_patches)])
        if args.obj_detec:
            pm_preds = [p for p in pm_preds if p[1]['labels'].size > 0]
            if len(pm_preds) > 0:
                preds = model.adjust_bboxes_to_full_img(im, pm_preds, patcher.patch_size)
                preds = [model.merge_preds(preds)]
            else:
                preds = []
            if args.save_dets:
                if gt is None:
                    raise Exception("No ground truth available")
                model.save_dets(img_id, preds[0], gt=gt)
            if len(pm_preds) > 0:
                preds = [model.filter_preds_by_conf(preds[0], args.conf_thresh)]
            title = f'Predictions with confidence > {args.conf_thresh}'
            plot_name = f'{file}_01_conf_{args.conf_thresh}{ext}'
        elif args.classif:
            neighbors = {p['patch_path']: patcher.neighboring_patches(p, im.shape, d=1) for p in pm}
            #DEBUG NEIGHBORS:
            #[print("Error", p, "does not seem correct") for neigh in neighbors.values() for p in neigh if p not in [a['patch_path'] for a in pm]]
            preds = {pm['patch_path']: pred for pm, pred in pm_preds}
            preds = model.prediction_validation(preds, neighbors)
            title = f'Body localization'
            plot_name = f'{file}_body_loc_{ext}'
        if args.draw_patches:
            DrawHelper().draw_patches(im, pm, patcher.patch_size)
        if not args.no_graphs:
            model.show_preds(im, preds, title=f'{title} for {file}{ext}', fname=plot_name)

if __name__ == '__main__':
    main()
