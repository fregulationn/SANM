# SANM
End-to-end Spatial Attention Network with Feature Mimicking for Head Detection



This code repo is built on [faster-rcnn.pytorch](https://github.com/jwyang/faster-rcnn.pytorch). 

## Installation and Preparation

Firstly, clone the code

```
https://github.com/fregulationn/SANM.git
```

Then, create a folder:
```
cd faster-rcnn.pytorch && mkdir data
```

### prerequisites

* Python 3.6
* Pytorch 1.0
* CUDA 8.0 or higher

### Data Preparation

* **Brainwash**: The dataset is in VOC format. Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare  datasets. Actually, you can refer to any others. After downloading the data, create softlinks in the folder data.
* **SCUT HEAD**:  Same as Brainwash. 
* **NUDT HEAD**:  This is a private database, please contact the author if needed.

## Train and Test
Before training, the cuda libs are required to compiled by:

```
cd lib

sh make.sh

```

We have provided train&test code for SANM. Just run:

```
sh train_test.sh
```

Download trained model from [Google dirve](https://drive.google.com/open?id=1z0UemgZo1-8ZzAu_vRC0_tQ2Cj8Q-nGR).


## Citation

    @article{jjfaster2rcnn,
        Author = {Jianwei Yang and Jiasen Lu and Dhruv Batra and Devi Parikh},
        Title = {A Faster Pytorch Implementation of Faster R-CNN},
        Journal = {https://github.com/jwyang/faster-rcnn.pytorch},
        Year = {2017}
    }
    
    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }