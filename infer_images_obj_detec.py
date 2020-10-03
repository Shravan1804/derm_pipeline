import os
import cv2
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as plt_patches

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


from PatchExtractor import PatchExtractor, DrawHelper
from ObjDetecPatchSamplerDataset import ObjDetecPatchSamplerDataset
import common

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


class ObjDetecModel:
    def __init__(self, classes, model_path, output_dir, use_cpu):
        self.model_path = model_path
        self.output_dir = output_dir
        self.device = torch.device('cuda') if not use_cpu and torch.cuda.is_available() else torch.device('cpu')
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

    classes = ['__background__', 'pustule', 'brown_spot'] if args.classes is None else args.classes
    model = ObjDetecModel(classes, args.model, args.out_dir, args.cpu)

    patcher = PatchExtractor(args.ps)
    for img_id, img_path in enumerate(img_list):
        print(f"Image {img_id}/{len(img_list)}: {img_path}")
        file, ext = os.path.splitext(os.path.basename(img_path))
        im = cv2.cvtColor(patcher.load_image(img_path), cv2.COLOR_BGR2RGB)

        gt = ObjDetecModel.get_img_gt(img_path)
        if args.out_dir is not None and gt is None:
            raise Exception("No ground truth available, but user requested to save dets")
        if args.show_gt:
            model.show_preds(im, [gt], title=f'Ground Truth for {file}{ext}', fname=f'{file}_00_gt{ext}')

        patch_maps = patcher.patch_grid(img_path, im)
        b_pms = common.batch_list(patch_maps, args.bs)

        preds = []
        for batch in tqdm(b_pms) if len(b_pms) > 1 else b_pms:
            patches = [PatchExtractor.extract_patch(im, pm) for pm in batch]
            batch_preds = model.predict_imgs(patches)
            preds.append(batch_preds)
            # TODO handle args.heatmap

        pm_preds = [p for p in zip(patch_maps, preds) if p[1]['labels'].size > 0]
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

    if not args.cpu:
        common.maybe_set_gpu(args.gpuid, args.num_gpus)

    common.time_method(main, args)

