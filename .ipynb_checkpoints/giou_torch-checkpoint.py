
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
