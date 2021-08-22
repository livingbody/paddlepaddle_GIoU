import paddle
import torch
try:
    from giou_paddle import giou_p
except:
    print("The code not found! Please rename your script to 'giou_paddle.py'")
    import sys
    sys.exit(1)
from giou_torch import giou_t

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

torch_bboxes1 = torch.FloatTensor(bboxes1)
torch_bboxes2 = torch.FloatTensor(bboxes2)
torch_overlaps = giou_t(torch_bboxes1, torch_bboxes2)

diff = (paddle_overlaps - torch_overlaps).sum()
if diff == 0:
    print('success!')
else:
    print('fail...')
    print('paddle result is {}'.format(paddle_overlaps))
    print('torch result is {}'.format(torch_overlaps))

