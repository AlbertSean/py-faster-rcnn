## 说明
本文列举py-faster-rcnn代码在使用中常见错误和解决办法。

**error 1**
`AttributeError: 'module' object has no attribute 'text_format'`
训练期间发生。
是因为protobuf的python接口升级导致的。

解决办法：
在lib/fast_rcnn/train.py中添加：
```python
import google.protobuf.text_format 
```

**error 2**
`TypeError: slice indices must be integers or None or have an __index__ method`
训练期间发生。
是因为新版numpy不支持浮点数作为索引。

解决办法：把浮点数显式转化为int型：
修改`~/my_faster_rcnn/lib/rpn/proposal_target_layer.py`，转到123行，原来内容：
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

**error 3**
`TypeError: 'numpy.float64' object cannot be interpreted as an index`
训练期间发生。
也是因为新版numpy不支持浮点数作为索引。

解决办法：
`~/my_faster_rcnn/lib/roi_data_layer/minibatch.py`第26行行尾，添加`.astype(np.int)`，得到：
```python
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```

`~/my_faster_rcnn/lib/datasets/ds_utils.py`第12行行尾，添加`.astype(np.int)`，得到：
```python
hashes = np.round(boxes * scale).dot(v).astype(np.int)
```

`~/my_faster_rcnn/lib/fast_rcnn/test.py`第129行行尾，添加`.astype(np.int)`，得到：
```
hashes = np.round(blobs['rois'] * cfg.DEDUP_BOXES).dot(v).astype(np.int)
```

`~/my_faster_rcnn/lib/rpn/proposal_target_layer.py`第60行行尾，添加`.astype(np.int)`，得到：
```
fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image).astype(np.int)
```