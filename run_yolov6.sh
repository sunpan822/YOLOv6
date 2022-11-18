# !/bin/bash

python /project/train/src_repo/split_yolov6.py
cd /project/train/src_repo/YOLOv6


# cp /project/train/models/exp/weights/last_ckpt.pt /project/train/models/exp/weights/best_ckpt.pt
python /project/train/src_repo/YOLOv6/tools/train.py --batch 32 --epochs 19 --data /project/train/src_repo/YOLOv6/data/objects.yaml --conf /project/train/src_repo/YOLOv6/configs/yolov6s_finetune.py  --img-size 640 --output-dir /project/train/models/ --workers 4 --resume /project/train/models/exp/weights/last_ckpt.pt

# python /project/train/src_repo/YOLOv6/tools/train.py --batch 32 --epochs 100 --data /project/train/src_repo/YOLOv6/data/objects.yaml --conf /project/train/src_repo/YOLOv6/configs/yolov6s_finetune.py  --img-size 640 --output-dir /project/train/models/ --workers 4 