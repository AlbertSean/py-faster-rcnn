### 训练自己的数据集

#### 下载本项目

下载本项目。并自行把各种预训练的.v2.caffemodel下载下来。参考原版py-faster-rcnn的说明。

#### 先准备数据集

比如放在`/home/chris/data/traffic_sign`目录：
```
├── Annotations
├── classes.txt
├── ImageSets
├── JPEGImages
└── 数据集说明.txt
```
其中classex.txt的每行是一个待检测的目标类别，不需要引号

Annotations, ImageSets, JPEGImages这3个目录，和VOC 2007数据集格式一样即可

需要确保`ImageSets/Main/xxx.txt`的存在: xxx通常是`train, val, trainval, test`等。这个xxx的取值要和后续训练脚本中指定的`TRAIN_IMDB`和`TEST_IMDB`取值的最后一段保持一致。

#### 修改配置文件

1. 数据集所在父目录

`experiments/cfg/faster_rcnn_end2end.yml`，修改`DATA_DIR`为你的数据集目录的父目录。


2. VOCdevkit目录

默认假定了VOCdevkit放在了/home/chris/data/VOCdevkit，可以在`experiments/cfg/faster_rcnn_end2end.sh`添加`VOCdevkit: /path/to/your/VOCdevkit`

或者`lib/fast_rcnn/config.py`中修改`__C.DEVKIT_DIR`的值

3. 训练脚本增加数据集

在`py-faster-rcnn/experiments/scripts/faster_rcnn_end2end.sh`等脚本中，添加数据集的case选项，如：
```bash
case $DATASET in
  traffic_sign)
    TRAIN_IMDB="你的数据集名_train"
    TEST_IMDB="你的数据集名_val"
    PT_DIR="你的数据集名"
    ITERS=1000
    ;;
```
`train`和`val`是用来取数据集子集的，对应着从`你的数据集名/ImageSets/Main/{train.txt,val.txt}`中取出。

#### 修改神经网络配置文件（各种prototxt)

根据上一步设定的`PT_DIR`准备。详细路径如：

`py-faster-rcnn/models/${PT_DIR}/${NET}/faster_rcnn_end2end/{solver.prototxt,train.prototxt,test.prototxt}`

其中`NET`是上一步的脚本的输入参数所设定,比如VGG16

很直接的一个做法是，从现有网络结构拷贝：
```
cp -R models/pascal_voc models/traffic_sign
```

**注意，一定记得修改solver.prototxt中的train.prototxt的路径！**
```
vim models/traffic_sign/VGG16/faster_rcnn_end2end/solver.prototxt

#修改第一行的路径，把默认的pascal_voc换成你的路径，例如traffic_sign
```

**如有必要，修改网络类别数**

在train.prototxt中：
```
input-data层的`num_classes`，为类别数+1 （1个背景类，下同）
roi-data层的`num_classes`，为类别数+1
cls_score层的`num_output`，为类别数+1
bbox_pred层的`num_output`，为(类别数+1)*4， 4表示一个bbox的4个坐标值
```

在test.prototxt中

**如有必要，修改anchor数**

```
rpn_cls_prob_reshape层的第二个`dim`: 2*anchor数量（2表示bg/fg，背景和前景做二分类,下同）
rpn_cls_score层的`num_output`: 2*anchor数量
```
同时，python代码中也要修改这个anchor数。具体要自己看下。

#### 开始训练

通常训练会耗时很久。强烈建议开启tmux执行任务，这样可以_断开_“执行训练所使用的那个shell”，等过一段时间后再连接上并查看它。
```
sudo apt install tmux

tmux new -s py-faster-rcnn-train  #建立tmux新会话，并指定其名字。

...   #开启各种耗时的命令、任务脚本

# ctrl+b是tmux各种组合键的前缀
# ctrl+b d  关闭当前shell

tmux a -t py-faster-rcnn-train  #重新连接指定的tmux会话

tmux ls   #查看tmux会话列表
```


关键训练脚本：
```
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG_CNN_M_1024 traffic_sign --ext .png
```
最后的两个参数用来指定训练图片后缀。不指定这两个参数的话默认是jpg格式


### 参考

https://github.com/andrewliao11/py-faster-rcnn-imagenet


### Disclaimer

The official Faster R-CNN code (written in MATLAB) is available [here](https://github.com/ShaoqingRen/faster_rcnn).
If your goal is to reproduce the results in our NIPS 2015 paper, please use the [official code](https://github.com/ShaoqingRen/faster_rcnn).

This repository contains a Python *reimplementation* of the MATLAB code.
This Python implementation is built on a fork of [Fast R-CNN](https://github.com/rbgirshick/fast-rcnn).
There are slight differences between the two implementations.
In particular, this Python port
 - is ~10% slower at test-time, because some operations execute on the CPU in Python layers (e.g., 220ms / image vs. 200ms / image for VGG16)
 - gives similar, but not exactly the same, mAP as the MATLAB version
 - is *not compatible* with models trained using the MATLAB code due to the minor implementation differences
 - **includes approximate joint training** that is 1.5x faster than alternating optimization (for VGG16) -- see these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more information

# *Faster* R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

By Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun (Microsoft Research)

This Python implementation contains contributions from Sean Bell (Cornell) written during an MSR internship.

Please see the official [README.md](https://github.com/ShaoqingRen/faster_rcnn/blob/master/README.md) for more details.

Faster R-CNN was initially described in an [arXiv tech report](http://arxiv.org/abs/1506.01497) and was subsequently published in NIPS 2015.

### License

Faster R-CNN is released under the MIT License (refer to the LICENSE file for details).

### Citing Faster R-CNN

If you find Faster R-CNN useful in your research, please consider citing:

    @inproceedings{renNIPS15fasterrcnn,
        Author = {Shaoqing Ren and Kaiming He and Ross Girshick and Jian Sun},
        Title = {Faster {R-CNN}: Towards Real-Time Object Detection
                 with Region Proposal Networks},
        Booktitle = {Advances in Neural Information Processing Systems ({NIPS})},
        Year = {2015}
    }

### Contents
1. [Requirements: software](#requirements-software)
2. [Requirements: hardware](#requirements-hardware)
3. [Basic installation](#installation-sufficient-for-the-demo)
4. [Demo](#demo)
5. [Beyond the demo: training and testing](#beyond-the-demo-installation-for-training-and-testing-models)
6. [Usage](#usage)

### Requirements: software

1. Requirements for `Caffe` and `pycaffe` (see: [Caffe installation instructions](http://caffe.berkeleyvision.org/installation.html))

  **Note:** Caffe *must* be built with support for Python layers!

  ```make
  # In your Makefile.config, make sure to have this line uncommented
  WITH_PYTHON_LAYER := 1
  # Unrelatedly, it's also recommended that you use CUDNN
  USE_CUDNN := 1
  ```

  You can download my [Makefile.config](https://dl.dropboxusercontent.com/s/6joa55k64xo2h68/Makefile.config?dl=0) for reference.
2. Python packages you might not have: `cython`, `python-opencv`, `easydict`
3. [Optional] MATLAB is required for **official** PASCAL VOC evaluation only. The code now includes unofficial Python evaluation code.

### Requirements: hardware

1. For training smaller networks (ZF, VGG_CNN_M_1024) a good GPU (e.g., Titan, K20, K40, ...) with at least 3G of memory suffices
2. For training Fast R-CNN with VGG16, you'll need a K40 (~11G of memory)
3. For training the end-to-end version of Faster R-CNN with VGG16, 3G of GPU memory is sufficient (using CUDNN)

### Installation (sufficient for the demo)

1. Clone the Faster R-CNN repository
  ```Shell
  # Make sure to clone with --recursive
  git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
  ```

2. We'll call the directory that you cloned Faster R-CNN into `FRCN_ROOT`

   *Ignore notes 1 and 2 if you followed step 1 above.*

   **Note 1:** If you didn't clone Faster R-CNN with the `--recursive` flag, then you'll need to manually clone the `caffe-fast-rcnn` submodule:
    ```Shell
    git submodule update --init --recursive
    ```
    **Note 2:** The `caffe-fast-rcnn` submodule needs to be on the `faster-rcnn` branch (or equivalent detached state). This will happen automatically *if you followed step 1 instructions*.

3. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/lib
    make
    ```

4. Build Caffe and pycaffe
    ```Shell
    cd $FRCN_ROOT/caffe-fast-rcnn
    # Now follow the Caffe installation instructions here:
    #   http://caffe.berkeleyvision.org/installation.html

    # If you're experienced with Caffe and have all of the requirements installed
    # and your Makefile.config in place, then simply do:
    make -j8 && make pycaffe
    ```

5. Download pre-computed Faster R-CNN detectors
    ```Shell
    cd $FRCN_ROOT
    ./data/scripts/fetch_faster_rcnn_models.sh
    ```

    This will populate the `$FRCN_ROOT/data` folder with `faster_rcnn_models`. See `data/README.md` for details.
    These models were trained on VOC 2007 trainval.

### Demo

*After successfully completing [basic installation](#installation-sufficient-for-the-demo)*, you'll be ready to run the demo.

To run the demo
```Shell
cd $FRCN_ROOT
./tools/demo.py
```
The demo performs detection using a VGG16 network trained for detection on PASCAL VOC 2007.

### Beyond the demo: installation for training and testing models
1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
    Using symlinks is a good idea because you will likely want to share the same PASCAL dataset installation between multiple projects.
5. [Optional] follow similar steps to get PASCAL VOC 2010 and 2012
6. [Optional] If you want to use COCO, please see some notes under `data/README.md`
7. Follow the next sections to download pre-trained ImageNet models

### Download pre-trained ImageNet models

Pre-trained ImageNet models can be downloaded for the three networks described in the paper: ZF and VGG16.

```Shell
cd $FRCN_ROOT
./data/scripts/fetch_imagenet_models.sh
```
VGG16 comes from the [Caffe Model Zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo), but is provided here for your convenience.
ZF was trained at MSRA.

### Usage

To train and test a Faster R-CNN detector using the **alternating optimization** algorithm from our NIPS 2015 paper, use `experiments/scripts/faster_rcnn_alt_opt.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_alt_opt.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

("alt opt" refers to the alternating optimization training algorithm described in the NIPS paper.)

To train and test a Faster R-CNN detector using the **approximate joint training** method, use `experiments/scripts/faster_rcnn_end2end.sh`.
Output is written underneath `$FRCN_ROOT/output`.

```Shell
cd $FRCN_ROOT
./experiments/scripts/faster_rcnn_end2end.sh [GPU_ID] [NET] [--set ...]
# GPU_ID is the GPU you want to train on
# NET in {ZF, VGG_CNN_M_1024, VGG16} is the network arch to use
# --set ... allows you to specify fast_rcnn.config options, e.g.
#   --set EXP_DIR seed_rng1701 RNG_SEED 1701
```

This method trains the RPN module jointly with the Fast R-CNN network, rather than alternating between training the two. It results in faster (~ 1.5x speedup) training times and similar detection accuracy. See these [slides](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0) for more details.

Artifacts generated by the scripts in `tools` are written in this directory.

Trained Fast R-CNN networks are saved under:

```
output/<experiment directory>/<dataset name>/
```

Test outputs are saved under:

```
output/<experiment directory>/<dataset name>/<network snapshot name>/
```
