# GIoU模块转写及对齐
## GIoU相关资料
原论文： https://giou.stanford.edu/GIoU.pdf

相关知识介绍： https://zhuanlan.zhihu.com/p/94799295
# 作业要求
使用PaddlePaddle实现GIoU

使用/home/aistudio/work/test_giou.py脚本，对齐PaddlePaddle实现与Pytorch实现，即运行test_giou.py后，出现success
# 提示
Pytorch参考实现位于/home/aistudio/work/giou_torch.py，另外AIStudio上无法直接运行Pytorch脚本，对齐时需要使用自己的环境。

测试case位于/home/aistudio/work/test_giou.py

PaddlePaddle实现脚本需要实现在/home/aistudio/work/giou_paddle.py中


```python

import torch

def giou_t(bboxes1, bboxes2, eps=1e-6):
    """Calculate overlap between two set of bboxes.
    Note:

    calculate the GIoU between each aligned pair of bboxes1 and bboxes2.
    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format or empty.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.
    Returns:
        Tensor: shape (m, )
    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0., 0., 10., 10.],
        >>>     [10., 10., 20., 20.],
        >>>     [32., 32., 38., 42.],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0., 0., 10., 20.],
        >>>     [0., 10., 10., 19.],
        >>>     [10., 10., 20., 20.],
        >>> ])
        >>> overlaps = giou(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, )
    """

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    assert rows == cols

    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt).clamp(min=0, max=None)
    overlap = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - overlap
    enclosed_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])
    enclosed_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union

    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clamp(min=0, max=None)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious.numpy()

```


```python
!python giou_torch.py
```


```python
bboxes1 = [
    [0., 0., 10., 10.],
    [10., 10., 20., 20.],
    [32., 32., 38., 42.],
]

bboxes2 = [
    [0., 0., 10., 10.],
    [0., 10., 10., 19.],
    [10., 10., 20., 20.],
]


torch_bboxes1 = torch.tensor(bboxes1)
torch_bboxes2 = torch.tensor(bboxes2)
torch_overlaps = giou_t(torch_bboxes1, torch_bboxes2)
print(torch_overlaps)
```

    [ 1.        -0.05      -0.8214286]
    


```python
import paddle

def giou_p(bboxes1, bboxes2, eps=1e-6):
    rows = bboxes1.shape[-2]
    cols = bboxes2.shape[-2]
    assert rows == cols
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
        bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
        bboxes2[:, 3] - bboxes2[:, 1])

    lt = paddle.maximum(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
    rb = paddle.minimum(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

    wh = (rb - lt).clip(min=0, max=None)
    overlap = wh[:, 0] * wh[:, 1]

    union = area1 + area2 - overlap
    enclosed_lt = paddle.minimum(bboxes1[:, :2], bboxes2[:, :2])
    enclosed_rb = paddle.maximum(bboxes1[:, 2:], bboxes2[:, 2:])

    eps = paddle.to_tensor([1e-6])
    union = paddle.maximum(union, eps)
    ious = overlap / union

    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0, max=None)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    enclose_area = paddle.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area

    return gious.numpy()

bboxes1 = [
    [0., 0., 10., 10.],
    [10., 10., 20., 20.],
    [32., 32., 38., 42.],
]

bboxes2 = [
    [0., 0., 10., 10.],
    [0., 10., 10., 19.],
    [10., 10., 20., 20.],
]


paddle_bboxes1 = paddle.to_tensor(bboxes1)
paddle_bboxes2 = paddle.to_tensor(bboxes2)
paddle_overlaps = giou_p(paddle_bboxes1, paddle_bboxes2)

print(paddle_overlaps)
```

    [ 1.        -0.05      -0.8214286]
    


```python
diff = (paddle_overlaps - torch_overlaps).sum()
if diff == 0:
    print('success!')
else:
    print('fail...')
    print('paddle result is {}'.format(paddle_overlaps))
    print('torch result is {}'.format(torch_overlaps))
```

    success!
    


```python
!python test_giou.py
```

    success!
    

    W0822 23:50:04.793823 19540 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 6.1, Driver API Version: 11.4, Runtime API Version: 10.2
    W0822 23:50:04.799830 19540 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    F:\Users\livingbody\miniconda3\envs\torch\lib\site-packages\win32\lib\pywintypes.py:2: DeprecationWarning: the imp module is deprecated in favour of importlib; see the module's documentation for alternative uses
      import imp, sys, os
    


```python

```
