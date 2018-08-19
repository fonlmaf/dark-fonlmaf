# dark-fonlmaf
## Introduction
This repository contains the proposed optimization techniques for real-time object detection networks and and the solutions for 4-channel input layer.

We try to deploy a object detection mpodel based on [YOLOv2-Tiny detector](https://pjreddie.com/darknet/yolov2/), which consists of a backbone network for feature extraction and a detection network for candidate bounding box generation.
In order to address the problem of tiny objects, occlusions and distractions from the provided data set, various optimizations are implemented on the network architecture for both training and inference.
We use [Feature Pyramid Network](https://arxiv.org/abs/1612.03144v2) with a top-down architecture for building high-level semantic feature maps at multiple scales.
In the focus layer, the [Focal Loss](https://arxiv.org/abs/1708.02002) is applied to  mitigate the imbalance between the single ground truth box and the candidate boxes at training phase,  thereby partially resolving occlusions and distractions. We remove the classification function on this layer, and there are only locations and confidence information in the anchor box.

Moreover, we propose an extended data layer, which contains 4 channels instead of 3 channels for orignal RGB images. It derives from the deep learning architecture on hardware platform such as FPGA and ASIC, in which the parallel convolution computing unit are normally designed with even number of channels. The extended data layer eliminates the resource waste on the first layer, and meanwhile brings a slightly performacne improvement.

The inference is evaluated on NVIDIA Jetson TX2 platform. We used multithreading to by loading images and infering in parallel, which improved about 7 FPS.

**Note:**  

We develop three projects for different purposes in this repository. YOLOv2, tiny-YOLOv2 and YOLOv2_v_4 model are deployed respectively  to compare the performance of combined techniques.  

The performance of our model with all the proposed techniques is as follows:

| Accuracy (mean IoU) | Baseline Accuracy (mean IoU) | Speed (FPS on Jetson TX2)
|:-----:|:-----:|:-----:|
| 0.845 | 0.784 | ~24 |

## Installation

*Prerequisites:*
 * OpenCV
 * CUDA/cuDNN
 * Python2/Python2-Numpy
 
 1. Download and build the source code. `$PROJECT_ROOT` is the directory of project `dark-fonmaf-master`
 ```Shell
git clone https://github.com/fonlmaf/dark-fonlmaf.git
cd $PROJECT_ROOT
make -j8
```

**Note:**
1. Our implementation is based on [Darknet framework](https://pjreddie.com/darknet/). You can also refer to the [installation guide](https://pjreddie.com/darknet/install/) of the original Darknet framework.
2. We implement the code for **GPU = 1**, **CUDNN = 1**, **OPENCV = 1** with single GPU mode.

## Training

1. Download the COCO raw dataset, annotations and metadata (about 120000 images and the correspounding labels in total).
```Shell
cd $TRAIN_ROOT/data
sh get_coco_data.sh
```

**Note:**
The training dataset and validation set are labeled in the donwloaded metadata file `$PROJECT_ROOT/coco/5k.txt` and `$PROJECT_ROOT/coco/trainvalno5k.txt`. The class names are stored in `$PROJECT_ROOT/coco/labels/`. You can also generate your own annotation file and class names file.

2. Modify train scripts and starts traing.

Train the optimized model with the proposed optimization techniques based on the tiny-YOLOv2 network.
```Shell
cd $PROJECT_ROOT/script
bash train_model.sh
```
Train the proposed 4-channel model based on the YOLOv2 network.
```Shell
cd $PROJECT_ROOT/script
bash train_yolov2_c_4.sh
```

## Validation
1. Validate the model trained by ourself. Configurate the project path in `$PROJECT_ROOT/script/valid_model.sh`.
```Shell
cd $PROJECT_ROOT/script
bash valid_model.sh
```
