## 说明
本分支fork自[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)，除了支持新版CUDA(8.0)、CuDNN(6.0)和numpy、protobuf外，仅将Python Layer中使用的`param_str_`改为`param_str`。

## 快速使用
```
git clone https://github.com/zchrissirhcz/py-faster-rcnn
cd py-faster-rcnn
git checkout rbg
git submodule update --init --recursive
cd caffe-fast-rcnn
make -j8 && make pycaffe
cd lib
make
```
以及遵循原版[py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn)中对于数据集、预训练模型的说明进行准备工作，然后训练或测试。

## 详细步骤
如果想自己体验一下本项目的建立过程，具体步骤如下：

#### 1 从github下载代码
```
# 切换到$HOME目录
cd ~

# 下载frcnn代码
git clone https://github.com/rbgirshick/py-faster-rcnn
cd py-faster-rcnn

# 下载frcnn依赖的rbg版caffe，其实有点老的
git submodule update --init --recursive
```

#### 2 修补caffe
可以用git来修改：
```
cd caffe-fast-rcnn  
git remote add caffe https://github.com/BVLC/caffe.git  
git fetch caffe  
git merge -X theirs caffe/master  
```
然后在合并之后注释掉`include/caffe/layers/python_layer.hpp`文件里的`self_.attr(“phase”) = static_cast(this->phase_)`


也可以手动修改：
```bash
# 下载最新版BVLC官方caffe，用于把相关层代码拷贝出来，替代caffe-fast-rcnn，使得能在cuda8等高版本cuda环境下使用
git clone https://github.com/BVLC/caffe caffe-BVLC --depth=1

# 为官方caffe和自己的caffe分别用变量存储其路径
BVLC_CAFFE=caffe-BVLC
MY_CAFFE=caffe-fast-rcnn

# 执行代码替换，使得能在高版本cuda,cudnn下使用caffe
cp $BVLC_CAFFE/include/caffe/layers/cudnn_relu_layer.hpp     $MY_CAFFE/include/caffe/layers/cudnn_relu_layer.hpp
cp $BVLC_CAFFE/include/caffe/layers/cudnn_sigmoid_layer.hpp  $MY_CAFFE/include/caffe/layers/cudnn_sigmoid_layer.hpp 
cp $BVLC_CAFFE/include/caffe/layers/cudnn_tanh_layer.hpp     $MY_CAFFE/include/caffe/layers/cudnn_tanh_layer.hpp
cp $BVLC_CAFFE/include/caffe/util/cudnn.hpp                  $MY_CAFFE/include/caffe/util/cudnn.hpp
cp $BVLC_CAFFE/src/caffe/layers/cudnn_relu_layer.cpp         $MY_CAFFE/src/caffe/layers/cudnn_relu_layer.cpp
cp $BVLC_CAFFE/src/caffe/layers/cudnn_relu_layer.cu          $MY_CAFFE/src/caffe/layers/cudnn_relu_layer.cu
cp $BVLC_CAFFE/src/caffe/layers/cudnn_sigmoid_layer.cpp      $MY_CAFFE/src/caffe/layers/cudnn_sigmoid_layer.cpp
cp $BVLC_CAFFE/src/caffe/layers/cudnn_sigmoid_layer.cu       $MY_CAFFE/src/caffe/layers/cudnn_sigmoid_layer.cu
cp $BVLC_CAFFE/src/caffe/layers/cudnn_tanh_layer.cpp         $MY_CAFFE/src/caffe/layers/cudnn_tanh_layer.cpp
cp $BVLC_CAFFE/src/caffe/layers/cudnn_tanh_layer.cu          $MY_CAFFE/src/caffe/layers/cudnn_tanh_layer.cu
```
并且将`caffe-fast-rcnn/src/caffe/layers/cudnn_conv_layer.cu`中的`_v3`找到并删除（一共出现两次，分别在85和103行，`CUDNN_CHECK(cudnnConvolutionBackwardFilter_v3(`去掉`_v3`)。

**修改Makefile.config**
```bash
cd caffe-fast-rcnn
vim Makefile.config
```

开启CUDNN:
```bash
USE_CUDNN := 1
```

提供显卡的60算力支持编译flag，并且去掉不支持的20等算力flag:
```bash
CUDA_ARCH := -gencode arch=compute_35,code=sm_35 \
         -gencode arch=compute_50,code=sm_50 \
         -gencode arch=compute_50,code=compute_50 \
         -gencode arch=compute_60,code=sm_60 \
         -gencode arch=compute_60,code=compute_60         
```

开启Caffe的Python接口编译开关：
```bash
WITH_PYTHON_LAYER := 1
```

添加hdf5头文件搜索路径（ubuntu16.04开始需要）:
```bash
# INCLUDE_DIRS路径上，追加/usr/include/hdf5/serial，因为hdf5这个包在ubuntu16.04换成hdf5/seiral这种用法了
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial
```

**修改Makefile**
```bash
vim Makefile
```

修改hdf5库名字（ubuntu14.04不需要，ubuntu16.04以及更新发行版需要):`hd5_hl` -> `hf5_serial_hl`, `hdf5`->`hdf5_serial`
```bash
LIBRARIES += glog gflags protobuf boost_system boost_filesystem m hdf5_serial_hl hdf5_serial
```

**修改Python接口(可选，建议使用)**
希望在使用python layer的时候，把`param_str_`改成`param_str`(去掉尾巴上的下划线)：
```bash
vim include/caffe/layers/python_layer.hpp
```
修改第27行：
```python
// self_.attr("param_str_") = bp::str( //原有
self_.attr("param_str") = bp::str(  //修改后的
```

**编译caffe**
```bash
make -j7
make pycaffe
```

#### 3　准备py-faster-rcnn需要的数据集、caffemodel文件
```bash
cd ~/py-faster-rcnn/data
ln -sf /opt/data/PASCAL_VOC/VOCdevkit2007 ./   #创建软链接。需要根据你的VOCdevkit实际路径进行替换
```

#### 4 编译py-faster-rcnn的lib
首先清空已有的.so文件，避免因为别人编译产生同名但不兼容的库文件：
```bash
cd ~/py-faster-rcnn/lib
find . -name '*.so' -exec rm {} \;
```

执行库文件编译：
```bash
make -j8
```

#### 5 修补py-faster-rcnn代码
**网络python层使用`param_str`替代`param_str_`(可选，推荐)**
如果caffe-fast-rcnn中有修改，则这里也要一起修改，保持两者接口名一致。
首先看看哪里出现了`param_str_`:
```bash
cd ~/py-faster-rcnn/lib
grep `param_str_` -Rni .
```

找到：
```bash
./roi_data_layer/layer.py:87:        layer_params = yaml.load(self.param_str_)
./rpn/proposal_layer.py:26:        layer_params = yaml.load(self.param_str_)
./rpn/proposal_target_layer.py:25:        layer_params = yaml.load(self.param_str_)
./rpn/anchor_target_layer.py:27:        layer_params = yaml.load(self.param_str_)
```

逐一修改上述文件上述行的`param_str_`为`param_str`

**运行demo**
运行demo,并不需要修改代码，只是为了给自己一个信号，现在的demo可以跑通了，可以松一口气了。
```
cd ~/chris_faster_rcnn
python tools/demo.py
```

接下来修复一系列python代码中出现的错误。如果不修复，具体的保存信息见[`error_solution_list.md`](https://github.com/zchrissirhcz/chris_faster_rcnn/tree/master/error_solution_list.md)文件

**修复error 1**
在lib/fast_rcnn/train.py中添加：
```python
import google.protobuf.text_format 
```

**修复error 2**
修改`~/chris_faster_rcnn/lib/rpn/proposal_target_layer.py`，转到123行，原来内容：
```python
for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
```
修改为：
```python
for ind in inds:
        ind = int(ind)
        cls = clss[ind]
        start = int(4 * cls)
        end = int(start + 4)
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
```

**修复error 3**
`~/chris_faster_rcnn/lib/roi_data_layer/minibatch.py`第26行行尾，添加`.astype(np.int)`，得到：
```python
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```

`~/chris_faster_rcnn/lib/datasets/ds_utils.py`第12行行尾，添加`.astype(np.int)`，得到：
```python
hashes = np.round(boxes * scale).dot(v).astype(np.int)
```

`~/chris_faster_rcnn/lib/fast_rcnn/test.py`第129行行尾，添加`.astype(np.int)`，得到：
```
hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v).astype(np.int)
```

`~/chris_faster_rcnn/lib/rpn/proposal_target_layer.py`第60行行尾，添加`.astype(np.int)`，得到：
```
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```

**限定anchor参数仅在一处设定（可选，推荐）**
原版代码中设定anchor参数实在过于灵活，反而容易出错，因为很容易忘记上次修改哪里，要检查的话要检查很多地方。

anchor的参数包括aspect ratio和scale两大系列参数，以及相关的`feat_stride`参数。
可以在训练的`prototxt`中指定，可以在`lib/rpn/generate_anchors.py`中设定，还可以在各个python layer如`lib/rpn/proposal_layer.py`和`lib/rpn/anchor_target_layer.py`中覆盖设定。现在，将其封锁在`lib/rpn/generate_anchors.py`中，则需要如下修改：

`lib/rpn/proposal_layer.py`，
```python
anchor_scales = layer_params.get('scales', (8, 16, 32))
self._anchors = generate_anchors(scales=np.array(anchor_scales))
```
修改为：
```python
# anchor_scales = layer_params.get('scales', (8, 16, 32))
# self._anchors = generate_anchors(scales=np.array(anchor_scales))
self._anchors = generate_anchors()
```

`lib/rpn/anchor_target_layer.py`,
```python
anchor_scales = layer_params.get('scales', (8, 16, 32))
self._anchors = generate_anchors(scales=np.array(anchor_scales))
```
修改为：
```python
# anchor_scales = layer_params.get('scales', (8, 16, 32))
self._anchors = generate_anchors(scales=np.array(anchor_scales))
self._num_anchors = generate_anchors()
```

并在`lib/rpn/generate_anchors.py`中添加代码来打印输出anchor的尺度参数和长款比参数：
```python
print('!!! ratios are:', ratios)
print('!!! scales are:', scales)
```

P.S. 经过上述修改后，如果需要修改anchor参数，则需要这几个步骤：

(1). 在`lib/rpn/generate_anchors.py`中修改默认值（因为获取anchor的代码都去掉了指定的anchor参数）

(2). 在训练和测试的prototxt中把`18`和`36`修改为新的anchor数量。

**尝试训练pascal_voc**
不是真的要训练它（真的训练它也行），意在保证训练能跑通。
```
./experiments/scripts/faster_rcnn_end2end.sh 0 VGG16 pascal_voc
```
