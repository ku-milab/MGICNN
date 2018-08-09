
# Multi-scale Gradual Itegration Convolutional Neural Network for False Positive Reduction in Pulmonary Nodule Detection

[Link][arXiv:1807.10581](https://arxiv.org/abs/1807.10581)

## Overview

#### Lung Nodula Analysis 2016 (LUNA16) 

This project for pulmonary nodule detection systems FP reduction in CT scans.

We use [LUNA16 challenge](https://luna16.grand-challenge.org/) dataset.

LUNA16 includes samples from 888 patients in the LIDC-IDRI open database, 

which contains annotations of the Ground Truth (GT) collected from 

the two-step annotation process by four experienced radiologists.

#### Motivation

Pulmonary nodules have a small size and various morphological features.

Therefore, it is difficult to classify using only the information of the nodule, so the adjacent region of nodules must be considered together.

The paper proposed a novel CNN architecture to consider the information with the surrounding area.

#### MGI-CNN 

Our model consist of Gradual Feature Extraction (GFE) and Multi-Stream Feature Integration (MSFI)

The GFE is a feature extraction method with a gradual strategy in multi-scale inputs.

For gradual strategy in cosider zoom-in and zoom-out strategy. 

* Zoom-in: Gradually extract information from a wide area to a narrow area.
* Zoom-out: Gradually extract information from a narrow area to a wide area.

The MSFI is consist with multi-stream feature representations and abstract-level feature integration.

#### Results

We participated in the competition and got the following CPMs:

- MILAB_ConcatCAD: rank 3 (2017.11.25)
- MILAB_RedCAD@: rank 8 (2017.11.25)

@:The submissions with '@'' used the initially provided 
list of nodule candidates computed using fewer candidate detection algorithms.

## Authors

Bum-Chae Kim, Jun-Sik Choi and prof. Heung-Il Suk*

*Corresponding author: hisuk@korea.ac.kr

Machine Intelligence Lab.,

Dept. Brain & Cognitive Engineering. 

Korea University, Seoul, Rep. of Korea.


## Thanks

JiSeok Yoon, thank you for your cooperation!


last update date: 2018.08.08.

