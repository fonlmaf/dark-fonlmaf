# dark-fonlmaf
## Introduction
This repository contains the proposed optimization techniques for real-time object detection networks and and the solutions for 4-channel input layer.

We try to deploy a object detection mpodel based on [YOLOv2-Tiny detector](https://pjreddie.com/darknet/yolov2/), which consists of a backbone network for feature extraction and a detection network for candidate bounding box generation.
In order to address the of tiny objects, occlusions and distractions from the provided data set, various optimizations are implemented on the network architecture for both training and inference.
In the focus layer, the [Focal Loss](https://arxiv.org/abs/1708.02002) is applied to  mitigate the imbalance between the single ground truth box and the candidate boxes at training phase,  thereby partially resolving occlusions and distractions. We remove the classification function on this layer, and there are only locations and confidence information in the anchor box.

Moreover, we propose an extended data layer, which contains 4 channls instead of 3 channels for orignal RGB images. It derives from the deep learning architecture on hardware platform such as FPGA and ASIC, in which the parallel convolution computing unit are normally designed with even number of channels. The extended data layer eliminates the resource waste on the first layer, and meanwhile brings a slightly performacne improvement.

The inference is evaluated on NVIDIA Jetson TX2 platform. We used multithreading to by loading images and infering in parallel, which improved about 7 FPS.

**Note:**  

We develop three projects for different purposes in this repository. YOLOv2, tiny-YOLOv2 and YOLOv2_v_4 model are deployed respectively  to compare the performance of combined techniques.  
