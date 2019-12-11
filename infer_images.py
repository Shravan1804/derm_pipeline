import os
import cv2
import sys
import math
import random
import argparse
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

import fastai.vision as fvision
from radam import *

class DrawHelper(object):
    def __init__(self, thickness=1, style='dotted', gap=10):
        self.thickness = thickness
        self.style = style
        self.gap = gap

    def drawline(self, im, pt1, pt2, color):
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** .5  # pythagoras hypotenuse
        pts = []
        for i in np.arange(0, dist, self.gap):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r) + .5)
            y = int((pt1[1] * (1 - r) + pt2[1] * r) + .5)
            p = (x, y)
            pts.append(p)

        if self.style == 'dotted':
            for p in pts:
                cv2.circle(im, p, self.thickness, color, -1)
        else:
            e = pts[0]
            for i, p in enumerate(pts):
                s = e
                e = p
                if i % 2 == 1:
                    cv2.line(im, s, e, color, self.thickness)

    def drawpoly(self, im, pts, color):
        e = pts[0]
        pts.append(pts.pop(0))
        for p in pts:
            s = e
            e = p
            self.drawline(im, s, e, color)

    def drawrect(self, im, pt1, pt2, color):
        pts = [pt1, (pt2[0], pt1[1]), pt2, (pt1[0], pt2[1])]
        self.drawpoly(im, pts, color)

    def test(self):
        im = np.zeros((800, 800, 3), dtype='uint8')
        patcher = PatchExtractor(256)
        patch_maps = patcher.img_as_grid_of_patches(im, 'test.jpg')
        patcher.draw_patches(im, patch_maps)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(im)
        plt.axis('off')
        plt.show()
        plt.close()


class PatchExtractor(object):
    def __init__(self, patch_size):
        self.patch_size = patch_size

    # TODO: filter grid_idx in in Datasets classes
    def img_as_grid_of_patches(self, im_arr, img_path):
        """Converts img into a grid of patches and returns the valid patches in the grid"""
        im_h, im_w = im_arr.shape[:2]
        if im_h < self.patch_size or im_w < self.patch_size:
            raise Exception(f'Error, patch size {self.patch_size} do not fit img shape {im_arr.shape}')
        step_h = self.patch_size - PatchExtractor.get_overlap(im_h, self.patch_size)
        # Don't forget + 1 in the stop argument otherwise the upper bound won't be included
        grid_h = np.arange(start=0, stop=1 + im_h - self.patch_size, step=step_h)
        step_w = self.patch_size - PatchExtractor.get_overlap(im_w, self.patch_size)
        grid_w = np.arange(start=0, stop=1 + im_w - self.patch_size, step=step_w)
        grid_idx = [self.get_patch_map(img_path, 0, a, b) for a in grid_h for b in grid_w]
        if not grid_idx:
            grid_idx = [self.get_patch_map(img_path, 0, 0, 0)]
        return grid_idx

    def maybe_resize(self, im_arr):
        """Resize img only if one of its dimensions is smaller than the patch size otherwise returns img unchanged"""
        h, w = im_arr.shape[:2]
        smallest = min(h, w)
        if smallest < self.patch_size:
            ratio = self.patch_size / smallest
            # resize new dim takes first w then h!
            return cv2.resize(im_arr, (max(self.patch_size, int(w * ratio)), max(self.patch_size, int(h * ratio))))
        else:
            return im_arr

    def load_img_from_disk(self, img_path):
        return self.maybe_resize(cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB))

    def get_patch_from_idx(self, im, id_h, id_w):
        return im[id_h:id_h + self.patch_size, id_w:id_w + self.patch_size]

    def get_patch_map(self, img_path, rotation, idx_h, idx_w):
        patch_name = PatchExtractor.get_patch_fname(img_path, idx_h, idx_w)
        return {'patch_path': patch_name, 'rotation': rotation, 'idx_h': idx_h, 'idx_w': idx_w}

    def draw_patches(self, img_arr, pm):
        draw = DrawHelper()
        for p in pm:
            s = (p['idx_w'], p['idx_h'])
            e = (p['idx_w'] + self.patch_size, p['idx_h'] + self.patch_size)
            draw.drawrect(img_arr, s, e, (255, 255, 255))
        return img_arr

    @staticmethod
    def get_overlap(n, div):
        remainder = n % div
        quotient = max(1, int(n / div))
        overlap = math.ceil((div - remainder) / quotient)
        return 0 if overlap == n else overlap

    @staticmethod
    def get_patch_suffix(idx_h, idx_w):
        return '_h' + str(idx_h) + '_w' + str(idx_w)

    @staticmethod
    def get_patch_fname(img_path, idx_h, idx_w):
        file, ext = os.path.splitext(os.path.basename(img_path))
        return file + PatchExtractor.get_patch_suffix(idx_h, idx_w) + ext


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

    def show_preds(self, img_arr, preds):
        raise NotImplementedError


class ClassifModel(CustomModel):
    def __init__(self, seed, classes, model_path, output_dir):
        super().__init__(seed, classes, model_path, output_dir)
        fvision.defaults.device = torch.device(self.device)

    def load_model(self, model_path):
        self.model = fvision.load_learner(os.path.dirname(model_path), os.path.basename(model_path))

    def prepare_img_for_inference(self, im):
        t = fvision.pil2tensor(im, dtype=im.dtype)  # converts to numpy tensor
        t = fvision.Image(t.float() / 255.)  # Convert to float
        return t

    def predict_imgs(self, ims):
        return [self.model.predict(self.convert_img(im)) for im in ims]

    def show_preds(self, img_arr, preds):
        print(preds)


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
        return [{k: v[np.where(pred['scores'] >= min_conf)] for k, v in pred.items()} for pred in preds
                if pred['labels'].size > 0]
        # if predictions is None: return None
        # pred_class, pred_boxes, pred_score = self.preds_to_cpu(predictions)
        # if not pred_score: return None  # nothing was detected
        # pred_t = [pred_score.index(x) for x in pred_score if x > thresh]
        # if not pred_t: return None  # no detection is confident enough
        # pred_t = pred_t[-1]
        # return pred_class[:pred_t + 1], pred_boxes[:pred_t + 1], pred_score[:pred_t + 1]

    def adjust_bboxes_to_full_img(self, patch_maps, preds):
        for i, pred in enumerate(preds):
            if len(pred['labels']) == 0:
                continue
            y, x = patch_maps[i]['idx_h'], patch_maps[i]['idx_w']
            pred['boxes'] += np.ones(pred['boxes'].shape) * np.array([x, y, x, y])
        return preds

    def show_preds(self, img_arr, preds, title='Predictions', fname='predictions.jpg'):
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12, 9))

        if len(preds) == 0:
            cv2.putText(img_arr, 'NOTHING DETECTED', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), thickness=3)
            ax.imshow(img_arr)
        else:
            preds = {k: np.concatenate(tuple(pred[k] for batch in preds for pred in batch
                                             if pred['labels'].size > 0), axis=0) for k in preds[0][0].keys()}
            classes = np.array(self.classes)
            pred_class = classes[preds['labels']]
            ax.imshow(img_arr)
            box_w = preds['boxes'][:, 2] - preds['boxes'][:, 0]
            box_h = preds['boxes'][:, 3] - preds['boxes'][:, 1]
            printed = {k: False for k in classes}
            for i, boxes in enumerate(preds['boxes']):
                color = self.bbox_colors[int(np.where(classes == pred_class[i])[0])]
                bbox = plt_patches.Rectangle((boxes[0], boxes[1]), box_w[i], box_h[i], linewidth=2, edgecolor=color,
                                             facecolor='none')
                ax.add_patch(bbox)
                if not printed[pred_class[i]]:
                    printed[pred_class[i]] = True
                    plt.text(boxes[0], boxes[3], s=pred_class[i], color='white', verticalalignment='top',
                         bbox={'color': color, 'pad': 0})
        plt.axis('off')
        plt.title(title)
        if self.output_dir is not None:
            plt.savefig(os.path.join(self.output_dir, fname))
        plt.show()
        plt.close()

def get_bbox_of_true_values(mask_with_condition):
    """Returns bbox coordinates: [xmin, ymin, xmax, ymax]. None if no objects"""
    pos = np.where(mask_with_condition)
    if pos[0].size == 0:  # no objects
        return None
    xmin = np.min(pos[1])
    xmax = np.max(pos[1])
    ymin = np.min(pos[0])
    ymax = np.max(pos[0])
    if xmin == xmax or ymin == ymax:    # Faulty object
        return None
    return [xmin, ymin, xmax, ymax]

def extract_mask_objs(mask):
    """Returns obj_count, obj_ids, obj_segm_areas, obj_bbox, obj_bbox_areas, obj_masks (without backgroud)"""
    obj_ids = np.unique(mask)
    # remove background
    obj_ids = obj_ids[1:]
    obj_count = len(obj_ids)
    if obj_count < 1:
        return None
    obj_masks = mask == obj_ids[:, None, None]
    boxes = [get_bbox_of_true_values(obj_masks[i]) for i in range(obj_count)]
    obj_del = [i for i, v in enumerate(boxes) if v is None]
    obj_count -= len(obj_del)
    if obj_count < 1:
        return None
    boxes = np.reshape(list(filter(None.__ne__, boxes)), (obj_count, 4))
    return obj_count, boxes

def get_img_gt(img_path):
    file, ext = os.path.splitext(os.path.basename(img_path))
    img_dir = os.path.dirname(img_path)
    root_dir = os.path.dirname(img_dir)
    masks = [cv2.imread(os.path.join(mask_path, file + '.png'), cv2.IMREAD_UNCHANGED) for mask_path
             in [os.path.join(root_dir, m) for m in sorted(os.listdir(root_dir)) if m.startswith('masks_')]]
    objs = list(zip(*filter(None.__ne__, map(extract_mask_objs, masks))))
    if not objs:  # check list empty
        return None
    objs[0] = tuple((i + 1) * np.ones(n_obj, dtype=np.int) for i, n_obj in enumerate(objs[0]))
    classes, boxes = (np.concatenate(val) for val in objs)
    return {'labels': classes, 'boxes': boxes}

def plot_img(img_arr, title, output_path):
    plt.figure()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_arr)
    plt.axis('off')
    plt.title(title)
    plt.savefig(output_path)
    plt.close()

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
    parser.add_argument('--save-output', action='store_true', help="Applies object detection model")
    parser.add_argument('--show-gt-only', action='store_true', help="Show gt of object detection dataset")
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

    if args.save_output:
        if args.out_dir is None:
            raise Exception("Error, --save-output is set but no --out-dir")
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
        model = ClassifModel(args.seed, args.classes, args.model, args.out_dir)
    else:
        raise Exception("Error, neither object detection nor classification was chosen")

    patcher = PatchExtractor(args.patch_size)
    for img_path in img_list:
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = patcher.load_img_from_disk(img_path)
        model.show_preds(im, [[get_img_gt(img_path)]], title=f'Ground Truth for {file}{ext}', fname=f'{file}_00_gt{ext}')
        if args.show_gt_only:
            continue
        print("Creating patches for", img_path)
        pm = patcher.img_as_grid_of_patches(im, img_path)
        # create batches
        b_pm = [pm[i:min(len(pm), i+args.batch_size)] for i in range(0, len(pm), args.batch_size)]
        b_patches = [(b, [patcher.get_patch_from_idx(im, pms['idx_h'], pms['idx_w']) for pms in b]) for b in b_pm]
        print("Applying model to patches")
        pm_preds = [(pms, model.predict_imgs(ims)) for pms, ims in tqdm(b_patches)]
        if args.draw_patches:
            patcher.draw_patches(im, pm)
            f_pm_preds = pm_preds
            title = 'Body Classification Prediction'
            plot_name = f'{file}_pred{ext}'
        if args.obj_detec:
            preds = [model.adjust_bboxes_to_full_img(pms, preds) for pms, preds in pm_preds]
            f_pm_preds = [model.filter_preds_by_conf(preds, args.conf_thresh) for preds in preds]
            f_pm_preds = [lst for lst in f_pm_preds if len(lst) > 0 and lst[0]['labels'].size > 0]
            title = f'Prediction with confidence greater than {args.conf_thresh}'
            plot_name = f'{file}_01_conf_{args.conf_thresh}{ext}'
        model.show_preds(im, f_pm_preds, title=f'{title} for {file}{ext}', fname=plot_name)

if __name__ == '__main__':
    main()
