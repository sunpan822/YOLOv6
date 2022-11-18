#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
import sys
import os.path as osp
import math
import torch
import json
import numpy as np
from PIL import ImageFont

sys.path.insert(1, "/project/train/src_repo/YOLOv6")

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.torch_utils import get_model_info

# ####### 参数设置
conf_thres = 0.5
iou_thres = 0.5
prob_thres = 0.5
#######
img_size = 480
weights = "/project/train/models/exp/weights/best_ckpt.pt"
device = '0'
stride = 32

img_source = "/home/data"
half = True
names = [ 'front_wear','front_no_wear','front_under_nose_wear','front_under_mouth_wear','mask_front_wear','mask_front_under_nose_wear','mask_front_under_mouth_wear','side_wear','side_no_wear','side_under_nose_wear',        'side_under_mouth_wear','mask_side_wear','mask_side_under_nose_wear','mask_side_under_mouth_wear','side_back_head_wear','side_back_head_no_wear','back_head','strap','front_unknown','side_unknown' ]


def init():
    global img_size, device, stride, half

    # Init model
    cuda = device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    model = DetectBackend(weights, device=device)

    stride = model.stride

    def font_check(font='/project/train/src_repo/YOLOv6/yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    font_check()

    def check_img_size(img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size, list) else [new_size] * 2

    def make_divisible(x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    img_size = check_img_size(img_size, s=stride)  # check image size

    def model_switch(model, img_size):
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()

        LOGGER.info("Switch model to deploy modality.")

    # Switch model to deploy status
    model_switch(model.model, img_size)

    # Half precision
    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    if device.type != 'cpu':
        model(
            torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup

    return model


def process_image(model, input_image=None, args=None, **kwargs):
    img_origin = input_image
    image = letterbox(input_image, img_size, stride=stride)[0]
    # Convert
    image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    image = torch.from_numpy(np.ascontiguousarray(image))
    image = image.half() if half else image.float()  # uint8 to fp16/32
    image /= 255  # 0 - 255 to 0.0 - 1.0

    img = image.to(device)
    if len(img.shape) == 3:
        img = img[None]
        # expand for batch dim
    pred_results = model(img)
    det = non_max_suppression(pred_results, conf_thres, 0.1)[0]    
    print(pred_results.size())
    # print(pred_results)

    # if len(det):
    #     print("===============================================")
    #     print(det)
    #     print("===============================================")

    gn = torch.tensor(img_origin.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    img_ori = img_origin.copy()

    # check image and font
    assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'

    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    fake_result = {}

    fake_result["algorithm_data"] = {
        "is_alert": False,
        "target_count": 0,
        "target_info": []
    }

    if len(det) > 0:
        det[:, :4] = rescale(img.shape[2:], det[:, :4], img_origin.shape).round()
        cnt = 0
        for *xyxy, conf, cls in reversed(det):

            if conf < prob_thres:
                continue
            fake_result["model_data"]['objects'].append({
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "width": int(xyxy[2] - xyxy[0]),
                "height": int(xyxy[3] - xyxy[1]),
                "confidence": float(conf),
                "name": names[int(cls)]
            })
            cnt += 1
            fake_result["algorithm_data"]["target_info"].append({
                "x": int(xyxy[0]),
                "y": int(xyxy[1]),
                "width": int(xyxy[2] - xyxy[0]),
                "height": int(xyxy[3] - xyxy[1]),
                "confidence": float(conf),
                "name": names[int(cls)]
            }
            )
            if cnt > 0:
                fake_result["algorithm_data"]["is_alert"] = True
                fake_result["algorithm_data"]["target_count"] = cnt
            else:
                fake_result["algorithm_data"]["target_info"] = []
            return json.dumps(fake_result, indent=4)


if __name__ == "__main__":
    from glob import glob
    import cv2
    import time

    # Test API
    image_names = glob('/home/data/**/*.jpg',recursive=True)
    # print(image_names)
    predictor = init()
    s = 0
    for image_name in image_names[:2]:
        print(image_name)
        img = cv2.imread(image_name)
        t1 = time.time()
        res = process_image(predictor, img)
        print(res)
        t2 = time.time()
        s += t2 - t1
    print(1 / (s / 100))
