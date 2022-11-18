# _*_ coding:utf-8 _*_
import shutil
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import cv2
import logging
from glob import glob
from glob import glob
from random import randint
import time
from multiprocessing import Process
from random import shuffle
from shutil import move, copy

'''
YOLO v5 
xml -> txt
'''
class2id = {
    "front_wear": 0,
    "front_no_wear": 1,
    "front_under_nose_wear": 2,
    "front_under_mouth_wear": 3,
    "mask_front_wear": 4,
    "mask_front_under_nose_wear": 5,
    "mask_front_under_mouth_wear": 6,
    "side_wear": 7,
    "side_no_wear": 8,
    "side_under_nose_wear": 9,
    "side_under_mouth_wear": 10,
    "mask_side_wear": 11,
    "mask_side_under_nose_wear": 12,
    "mask_side_under_mouth_wear": 13,
    "side_back_head_wear": 14,
    "side_back_head_no_wear": 15,
    "back_head": 16,
    "strap": 17,
    "front_unknown": 18,
    "side_unknown": 19
}


def convert(size, box):
    dw, dh = 1. / (size[0]), 1. / (size[1])
    x, y = (box[0] + box[1]) / 2.0 - 1, (box[2] + box[3]) / 2.0 - 1
    w, h = box[1] - box[0], box[3] - box[2]

    x, y, w, h = x * dw, w * dw, y * dh, h * dh
    x, y, w, h = min(x, 1.0), min(y, 1.0), min(w, 1.0), min(h, 1.0)
    return (x, y, w, h)


def convert_annotation(image_path, dest_path):
    in_file = open(image_path.replace('.jpg', '.xml'), encoding="utf-8")
    file_name = os.path.basename(image_path)
    out_file = open(os.path.join(dest_path, file_name.replace('.jpg', '.txt')), 'w')

    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    
    for obj in root.iter('object'):
        name = obj.find('name').text
        cls_id = class2id[name]
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), b, )
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def process_func(files: list, dest: str, mode: str):
    dest_img_path = os.path.join(dest, "images/train/")
    dest_label_path = os.path.join(dest, "labels/train/")
    if mode == "val":
        dest_img_path = os.path.join(dest, "images/val/")
        dest_label_path = os.path.join(dest, "labels/val/")
        
    for path in (dest_img_path,dest_label_path):
        if not os.path.exists(path):
            os.makedirs(path)

    for file in files:
        shutil.copy(file, dest_img_path)
        convert_annotation(file, dest_label_path)


def main(src_path: str = "/home/data",
         dest_path: str = "/project/train/src_repo/dataset",
         num_process=8,
         train_rate=0.8):
    img_files = glob(f"{src_path}/**/*.jpg", recursive=True)
    len_files = len(img_files)
    shuffle(img_files)

    train_files, val_files = img_files[:int(len_files * train_rate)], img_files[int(len_files * train_rate):]

    in_files = (train_files, val_files)
    modes = ("train", "val")

    for files, mode in zip(in_files, modes):
        pros = []
        per_len = len(files) // num_process
        for i in range(num_process):
            if i == num_process - 1:
                pros.append(Process(target=process_func, args=(files[i * per_len:], dest_path, mode)))
            pros.append(Process(target=process_func, args=(files[i * per_len:(i + 1) * per_len], dest_path, mode)))

        for pro in pros:
            pro.start()

        for pro in pros:
            pro.join()


if __name__ == '__main__':
    main()
