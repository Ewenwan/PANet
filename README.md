# 实例分割 Path Aggregation Network for Instance Segmentation

by [Shu Liu](http://shuliu.me), Lu Qi, Haifang Qin, [Jianping Shi](https://shijianping.me/), [Jiaya Jia](http://jiaya.me/).

Mask-Rcnn的改进版本，整体思路是提高信息流在网络中的传递效率。

第一个改进：

为了提高低层信息的利用率，加快低层信息的传播效率，提出了Bottom-up Path Augmentation；

第二个改进：

通常FPN在多层进行选anchors时，根据anchors的大小，将其分配到对应的层上进行分层选取。这样做很高效，但同时也不能充分利用信息了，提出了Adaptive Feature Pooling。

第三个改进：

为了提高mask的生成质量，作者将卷积-上采样和全连接层进行融合，提出了Fully-connected Fusion。


香港中文大学、北京大学、商汤科技、腾讯优图在CVPR 2018发表的一篇论文，提出了一个名为PANet的实例分割框架。
优化了Mask R-CNN中的信息传播，通过加速信息流、整合不同层级的特征，提高了生成预测蒙版的质量。
在未经大批量训练的情况下，就拿下了COCO 2017挑战赛实例分割任务的冠军。

[maskscoring_rcnn 2019改进版本](https://github.com/Ewenwan/maskscoring_rcnn)

### Introduction

This repository is for the CVPR 2018 Spotlight paper, '[Path Aggregation Network for Instance Segmentation](https://arxiv.org/abs/1803.01534)', which ranked 1st place of [COCO Instance Segmentation Challenge 2017](http://cocodataset.org/#detections-leaderboard) , 2nd place of [COCO Detection Challenge 2017](http://cocodataset.org/#detections-leaderboard) (Team Name: [UCenter](https://places-coco2017.github.io/#winners)) and 1st place of 2018 [Scene Understanding Challenge for Autonomous Navigation in Unstructured Environments](http://cvit.iiit.ac.in/scene-understanding-challenge-2018/benchmarks.php#instance) (Team Name: TUTU).

### Citation

If PANet is useful for your research, please consider citing:

    @inproceedings{liu2018path,
      author = {Shu Liu and
                Lu Qi and
                Haifang Qin and
                Jianping Shi and
                Jiaya Jia},
      title = {Path Aggregation Network for Instance Segmentation},
      booktitle = {Proceedings of IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
      year = {2018}
    }


### Disclaimer 

- The origin code was implemented based on the modified version of Caffe maintained by Sensetime Research. Due to several reasons, we could not release our origin code. 
- In this repository, we provide our re-implementation of PANet based on Pytorch. Note that our code is heavily based on [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). Thanks [Roy](https://github.com/roytseng-tw) for his great work!
- Several details, e.g., weight initialization and RPN joint training, in [Detectron](https://github.com/facebookresearch/Detectron) is fairly different from our origin implementation. In this repository, we simply follow Detectron because it achieves a better baseline than the codebase used in our paper. 
- In this repository, we test our code with BN layers in the backbone fixed and use GN in other part. We expect to achieve a better performance with Synchronized Batch Normalization Layer and train all parameter layers as what we have done in our paper. With those differences and a much better baseline, the improvement is **not** same as the one we reported. But we achieve a **better** performance than our origin implementation. 
- We trained with image batch size 16 using 8*P40. The performance should be similar with batch size 8.

### Installation

For environment requirements, data preparation and compilation, please refer to [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch).

WARNING: pytorch 0.4.1 is broken, see https://github.com/pytorch/pytorch/issues/8483. Use pytorch 0.4.0

### Usage

For training and testing, we keep the same as the one in [Detectron.pytorch](https://github.com/roytseng-tw/Detectron.pytorch). To train and test PANet, simply use corresponding config files. For example, to train PANet on COCO:

```shell
python tools/train_net_step.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml
```

To evaluate the model, simply use:

```shell
python tools/test_net.py --dataset coco2017 --cfg configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml --load_ckpt {path/to/your/checkpoint}
```

### Main Results


 Backbone     | Type   | Batch Size      | LR Schedules | Box AP | Mask AP | Download Links 
 :------------: |:------------: |:------------: |:------: | :-------: | :--------------:| :--------------:
 R-50-PANet (paper) | Faster | 16 | 1x | 39.2 | - | - 
 R-50-PANet | Faster | 16 | 1x | **39.8** | - | [model](https://drive.google.com/file/d/1_ahNQHY3D4mbsMWHR2FwmItBkLwYOrS4/view?usp=sharing) 
 R-50-PANet-2fc (paper) | Faster | 16 | 1x | 39.0 | - | - 
 R-50-PANet-2fc | Faster | 16 | 1x | **39.6** | - | [model](https://drive.google.com/file/d/1s-xm8GxHbmnt5M3gOMacXIRMvCGaDeRR/view?usp=sharing) 
 R-50-PANet (paper) | Mask| 16 | 2x | 42.1 | 37.8 | - 
 R-50-PANet | Mask | 16| 2x | **43.1** | **38.3** | [model](https://drive.google.com/file/d/1-pVZQ3GR6Aj7KJzH9nWoRQ-Lts8IcdMS/view?usp=sharing) 

Results on COCO 20017 *val* subset produced by this repository. In our paper, we used Synchronized Batch Normalization following all parameter layers. While in this repository, we fix BN layers in the backbone and use GN layers in other part. With the same set of hyper-parameters, e.g., multi-scales, this repository can produce better performance than that in our origin paper. We expect a better performance with Synchronized Batch Normalization Layer.

### Questions

Please contact 'liushuhust@gmail.com'
