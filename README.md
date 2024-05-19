# Adaptive Multiview Graph Convolutional Network for 3D Point Cloud Classification and Segmentation

## Authors
- Wanhao Niu
- Haowen Wang
- Chungang Zhuang

## Abstract
Point cloud classification and segmentation are crucial tasks for point cloud processing and have wide range of applications, such as autonomous driving and robot grasping. Some pioneering methods, including PointNet, VoxNet, DGCNN, etc., have made substantial advancements. However, most of these methods overlook the geometric relationships between points at large distances from different perspectives within the point cloud. This oversight constrains feature extraction capabilities and consequently limits any further improvements in classification and segmentation accuracy. To address this issue, we propose an adaptive multiview graph convolutional network (AM-GCN), which comprehensively synthesizes both the global geometric features of the point cloud and the local features within the projection planes of multiple views through an adaptive graph construction method. First, an adaptive rotation module in AM-GCN is proposed to predict a more favorable angle of view for projection. Then, a multi-level feature extraction network can flexibly be constructed by spatial-based or spectral-based graph convolution layers. Finally, AM-GCN is evaluated on ModelNet40 for classification, ShapeNetPart for part segmentation, ScanNetv2 and S3DIS for scene segmentation, which demonstrates the robustness of the AM-GCN with competitive performance compared with existing methods. It's worth noting that it performs state-of-the-art performance in many categories.

## Key Figures
![Figure 1](image/Figure1.svg)
![Figure 2](image/Figure2.svg)

## Publication
This paper has been accepted by IEEE Transactions on Cognitive and Developmental Systems (TCDS).

