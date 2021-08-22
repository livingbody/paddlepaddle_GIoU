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

    eps = paddle.to_tensor([eps])
    union = paddle.maximum(union, eps)
    ious = overlap / union

    # calculate gious
    enclose_wh = (enclosed_rb - enclosed_lt).clip(min=0, max=None)
    enclose_area = enclose_wh[:, 0] * enclose_wh[:, 1]
    enclose_area = paddle.maximum(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious.numpy()
