# MCFusion
**MCFusion: Infrared and Visible Image Fusion based Multiscale Receptive Field and Cross-Modal Enhanced Attention Mechanism**

## Abstract
Infrared and visible image fusion (IVIF) aims to generate a more comprehensive image for improving measurement and analysis accuracy. Existing Swin Transformer based methods effectively extract features from medium-large scale receptive fields. However, these nethods ignored the complementary role of features extracted from the small receptive field, resulting in the loss of essential fine information. To this end, we propose MCFusion, a dual-branch framework based on multiscale receptive field and cross-modal enhanced attention mechanism. MCFusion is composed of an attention-guided coarse branch (AGCB) and a fine branch (FineB). Firstly, AGCB and FineB are respectively designed to extract distinct yet complementary features from medium-large and small receptive fields. Secondly, a cross-modal enhanced attention mechanism is designed to enhance shared features across different modalities. Thirdly, a novel loss function is proposed to generate fused images, fully considering contrast, texture, and illumination intensity. The test results have demonstrated that our MCFusion is superior to other state-of-the-art methods, can notably enhance the detection and recognition of targets.

## Recommended Environment
 - [ ] python  3.8
 - [ ] torch  1.11.0
 - [ ] torchvision 0.12.0


## To Train
You need to download the [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/trainsets/MSRS/**.

Then, run `python main_train_mcfusion.py`.

## To Test
Download the test dataset from [**MSRS dataset**](https://github.com/Linfeng-Tang/MSRS), and put it in **./Dataset/testsets/MSRS/**. 

Then, run `python test_mcfusion.py`.

## Any Question
If you have any other questions about the code, please email `6213113119@stu.jiangnan.edu.cn`

