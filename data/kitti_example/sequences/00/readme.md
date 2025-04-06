## KITTI Odometry Dataset - 00 example

### General Information

This folder contains a part of the KITTI Odometry Dataset. It can be used for the testing and evaluation of range-sensor based mapping and reconstruction.

Specifically, this subset contains the 1st-100th LiDAR frame of the 00 sequence of KITTI odometry dataset as well as the corresponding pose and point-wise panoptic labels provided by Semantic KITTI.

For the original KITTI Dataset, please refer to: https://www.cvlibs.net/datasets/kitti/eval_odometry.php

For the original Semantic KITTI Dataset, please refer to: http://semantic-kitti.org/dataset.html



### Data

```
-- velodyne: the folder containing 100 frames of point cloud with *.bin format
-- labels: the folder containing 100 frames of the point-wise panoptic (instance-wise semantic) labels with *.label format
-- poses.txt: the pose under camera frame
-- calib.txt: the calibration transformation among different coordinate systems 
```



### Copyright

If you use this dataset in your project, please cite:

```
@inproceedings{geiger2012cvpr,
  author = {A. Geiger and P. Lenz and R. Urtasun},
  title = {{Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite}},
  booktitle = {Proc.~of the IEEE Conf.~on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3354--3361},
  year = {2012}
}

@inproceedings{behley2019iccv,
  author = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
  year = {2019}
}
```



### Contact

For further questions, please contact Yue Pan (yue.pan@igg.uni-bonn.de).