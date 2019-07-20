# A Pytorch implement of YOLOv3 with comments

This repo is copied from [Pytorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3). The goal of this repo is to learn how to build a YOLOv3 and add the detailed comments to the code. More details for the experimental results please refer to the original [repo](https://github.com/eriklindernoren/PyTorch-YOLOv3).

### The environment

- Python 3.6 :: Anaconda
- Pytorch >=0.4 
- COCO Dataset

### Usage

```sh
# Download pretrained weights
$ cd weights/
$ bash download_weights.sh

# Download COCO dataset
$ cd data/
$ bash get_coco_dataset.sh

# run the code
$ python3 train.py --model_def config/yolov3.cfg --data_config config/coco.data
$ python3 test.py --weights_path weights/yolov3.weights --img_size 416
$ python3 detect.py --image_folder data/samples
```

### To learn how to build a YOLOv3

I slightly simplify the original [repo](https://github.com/eriklindernoren/PyTorch-YOLOv3) and add the detailed comments on the main components of YOLO system. I mainly focus on the four parts of the YOLO:

- How to prepare the inputs? ==> utils/datasets.py
- How to bulid the targets for YOLO? ==> utils/utils.bulid_targets (function)
- How did the YOLO do with inputs and targets? ==> models.YOLOLayer (class)
- How to process the outputs of YOLO? ==> test.py, detect.py and utils/nms.py
