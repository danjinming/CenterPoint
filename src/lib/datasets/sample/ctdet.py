from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import imgaug as ia
import imgaug.augmenters as iaa
import random


class CTDetDataset(data.Dataset):
    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                        dtype=np.float32)
        return bbox

    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        # chartlet_dir = "/home/raid5/daming/HandDataMix/TrainImg/AnnImgMix"
        # select = random.random()
        # if select > 0.5:
        #   img_path = os.path.join(self.img_dir, file_name)
        # else:
        #   img_path = os.path.join(chartlet_dir, file_name)
        # img = self.imgs[index]
        path = img_path
        img = cv2.imread(path)  # BGR
        assert img is not None, 'Image Not Found ' + path
        h0, w0 = img.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # resize image to img_size
        if r < 1 or (self.augment and r != 1):  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 and not self.augment else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        return img, (h0, w0), img.shape[:2]  # img, hw_original, hw_resized

    def load_mosaic(self, index):
        # loads images in a mosaic
        self.augment = False
        self.img_size = 400
        labels4 = []
        s = self.img_size
        xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img_id = self.images[index]
            img, (h_src, w_src), (h, w) = self.load_image(index)

            # place img in img4
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            label_tmp = self.coco.loadAnns(ids=ann_ids)
            x = []
            for la in label_tmp:
                x_tmp = la['bbox']
                x_tmp = [x_tmp[0] / w_src, x_tmp[1] / h_src, x_tmp[2] / w_src, x_tmp[3] / h_src]
                x.append(x_tmp)
            x = np.array(x)
            labels = x.copy()

            if x.size > 0:  # Normalized xywh to pixel xyxy format
                labels[:, 0] = w * x[:, 0] + padw
                labels[:, 1] = h * x[:, 1] + padh
                labels[:, 2] = w * (x[:, 2] + x[:, 0]) + padw
                labels[:, 3] = h * (x[:, 3] + x[:, 1]) + padh
            labels4.append(labels)
        return img4, labels4

    def compute_iou(self, gt_box, b_box):
        '''
        计算iou
        :param gt_box: ground truth gt_box = [x0,y0,x1,y1]（x0,y0)为左上角的坐标（x1,y1）为右下角的坐标
        :param b_box: bounding box b_box 表示形式同上
        :return:
        '''
        width0 = gt_box[2] - gt_box[0]
        height0 = gt_box[3] - gt_box[1]
        width1 = b_box[2] - b_box[0]
        height1 = b_box[3] - b_box[1]
        max_x = max(gt_box[2], b_box[2])
        min_x = min(gt_box[0], b_box[0])
        width = width0 + width1 - (max_x - min_x)
        if width <= 0:
            width = 0
        max_y = max(gt_box[3], b_box[3])
        min_y = min(gt_box[1], b_box[1])
        height = height0 + height1 - (max_y - min_y)
        if height <= 0:
            height = 0

        interArea = width * height
        # boxAArea = width0 * height0
        # boxBArea = width1 * height1
        # iou = interArea / (boxAArea + boxBArea - interArea)
        return interArea

    def __getitem__(self, index):
        mosaic_pro = random.random()
        if mosaic_pro > 0:
            img_id = self.images[index]
            img, labels = self.load_mosaic(index)
            all_ann = []
            for da_label in labels:
                da_label = da_label.tolist()
                for da_l in da_label:
                    all_ann.append(da_l)
            num_objs = min(len(all_ann), self.max_objs)
        else:
            positive_aug = random.random()
            if positive_aug > 2:
                index1 = random.randint(0, self.num_samples - 1)
                # chartlet_dir = "/home/raid5/daming/HandDataMix/TrainImg/AnnImgMix"
                img_id = self.images[index]
                img_id1 = self.images[index1]

                file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
                file_name1 = self.coco.loadImgs(ids=[img_id1])[0]['file_name']

                path_num = random.random()
                img_path = os.path.join(self.img_dir, file_name)
                img_path1 = os.path.join(self.img_dir, file_name1)
                # if path_num > 0.5:
                #   img_path = os.path.join(chartlet_dir, file_name)

                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                ann_ids1 = self.coco.getAnnIds(imgIds=[img_id1])

                anns = self.coco.loadAnns(ids=ann_ids)
                anns1 = self.coco.loadAnns(ids=ann_ids1)

                img = cv2.imread(img_path)
                img1 = cv2.imread(img_path1)
                hand_num = len(anns1)
                if hand_num > 0:
                    for ann1 in anns1:
                        ran_id = random.randint(0, 26000)
                        hand_x = ann1['bbox'][0]
                        hand_y = ann1['bbox'][1]
                        hand_w = ann1['bbox'][2]
                        hand_h = ann1['bbox'][3]
                        temp = img1[hand_y:hand_y + hand_h, hand_x:hand_x + hand_w]
                        temp_h, temp_w, c = temp.shape
                        src_h, src_w, src_c = img.shape
                        for n in range(100):
                            min_src = min(src_w, src_h)
                            max_temp = max(temp_h, temp_w)
                            if (max_temp > 0.5 * min_src):
                                break
                            if (src_w < temp_w or src_h < temp_h):
                                break
                            x_tmp = random.randint(0, src_w - temp_w)
                            y_tmp = random.randint(0, src_h - temp_h)
                            src_rect = [x_tmp, y_tmp, x_tmp + temp_w, y_tmp + temp_h]
                            iou_all = 0
                            for gt in anns:
                                gt = [gt['bbox'][0], gt['bbox'][1], gt['bbox'][0] + gt['bbox'][2],
                                      gt['bbox'][1] + gt['bbox'][3]]
                                iou = self.compute_iou(gt, src_rect)
                                iou_all = iou_all + iou
                                # print(iou_all)
                                if iou_all == 0:
                                    img[y_tmp:y_tmp + temp_h, x_tmp:x_tmp + temp_w] = temp
                                    a = {'bbox': [x_tmp, y_tmp, temp_w, temp_h], 'category_id': 1}
                                    anns.append(a)
                                    break
                    num_objs = min(len(anns), self.max_objs)
            else:
                img_id = self.images[index]
                file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
                # daming_dir = "/home/raid5/daming/HandDataMix/TrainImg/AnnImgMix"
                img_path = os.path.join(self.img_dir, file_name)
                # img_path1 = os.path.join(daming_dir, file_name)
                ann_ids = self.coco.getAnnIds(imgIds=[img_id])
                anns = self.coco.loadAnns(ids=ann_ids)
                num_objs = min(len(anns), self.max_objs)
                img = cv2.imread(img_path)
                # daming_num = random.random()
                # if daming_num > 0.5:
                #   img = cv2.imread(img_path)
                # else:
                #   img = cv2.imread(img_path1)

        gray_pro = random.random()
        if gray_pro > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
                # s = s * np.random.choice(np.arange(0.3, 1.2, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        iaa_pro = random.random()
        if iaa_pro > 2:
            aug_seq = iaa.Sequential([iaa.MultiplyHueAndSaturation((0.5, 1.5), per_channel=True)])
            #   aug_seq = iaa.Sequential([
            #     iaa.Sometimes(
            #         0.5,
            #         iaa.GaussianBlur(sigma=(0, 0.5))
            #     ),
            #     iaa.LinearContrast((0.75, 1.5)),
            #     iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
            #     iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # ], random_order=True)
            inp, _ = aug_seq(image=inp, bounding_boxes=None)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        # ind is the center index, reg is the offset of center point in extracted feature maps
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            if mosaic_pro > 0:
                ann = all_ann[k]
                bbox = np.array([float(ann[0]), float(ann[1]), float(ann[2]), float(ann[3])], dtype=np.float32)
            else:
                ann = anns[k]
                bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = 0
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                # print("- h : ", h," - w : ", w)
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])
        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta

        return ret
