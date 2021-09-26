import paddle
from paddle.nn import functional as F
from typing import Union, Tuple
import numpy as np


def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')


def resize(input: paddle.Tensor, size: Union[int, Tuple[int, int]],
           interpolation: str = 'bilinear', align_corners: bool = False,
           side: str = "short") -> paddle.Tensor:
    r"""Resize the input paddle.Tensor to the given size.

    See :class:`~kornia.Resize` for details.
    """
    if not paddle.is_tensor(input):
        raise TypeError("Input tensor type is not a paddle.Tensor. Got {}"
                        .format(type(input)))

    input_size = h, w = input.shape[-2:]
    if isinstance(size, int):
        aspect_ratio = w / h
        size = _side_to_image_size(size, aspect_ratio, side)

    if size == input_size:
        return input

    return F.interpolate(input, size=size, mode=interpolation, align_corners=align_corners)


def _side_to_image_size(
    side_size: int, aspect_ratio: float, side: str = "short"
) -> Tuple[int, int]:
    if side not in ("short", "long", "vert", "horz"):
        raise ValueError(
            f"side can be one of 'short', 'long', 'vert', and 'horz'. Got '{side}'")
    if side == "vert":
        return side_size, int(side_size * aspect_ratio)
    elif side == "horz":
        return int(side_size / aspect_ratio), side_size
    elif (side == "short") ^ (aspect_ratio < 1.0):
        return side_size, int(side_size * aspect_ratio)
    else:
        return int(side_size / aspect_ratio), side_size


def center_crop(tensor: paddle.Tensor, size: Tuple[int, int],
                interpolation: str = 'bilinear',
                align_corners: bool = True) -> paddle.Tensor:
    r"""Crop the 2D images (4D tensor) at the center.

    Args:
        tensor (paddle.Tensor): the 2D image tensor with shape (B, C, H, W).
        size (Tuple[int, int]): a tuple with the expected height and width
          of the output patch.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.interpolate for details
    Returns:
        paddle.Tensor: the output tensor with patches.

    Examples:
        >>> input = paddle.tensor([[
                [1., 2., 3., 4.],
                [5., 6., 7., 8.],
                [9., 10., 11., 12.],
                [13., 14., 15., 16.],
             ]])
        >>> kornia.center_crop(input, (2, 4))
        tensor([[[ 5.0000,  6.0000,  7.0000,  8.0000],
                 [ 9.0000, 10.0000, 11.0000, 12.0000]]])
    """
    if not isinstance(tensor, paddle.Tensor):
        raise TypeError("Input tensor type is not a paddle.Tensor. Got {}"
                        .format(type(tensor)))
    if not isinstance(size, (tuple, list,)) and len(size) == 2:
        raise ValueError("Input size must be a tuple/list of length 2. Got {}"
                         .format(size))
    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # unpack input sizes
    dst_h, dst_w = size
    src_h, src_w = tensor.shape[-2:]

    # compute start/end offsets
    dst_h_half = dst_h / 2
    dst_w_half = dst_w / 2
    src_h_half = src_h / 2
    src_w_half = src_w / 2

    start_x = src_w_half - dst_w_half
    start_y = src_h_half - dst_h_half

    end_x = start_x + dst_w - 1
    end_y = start_y + dst_h - 1
    # [y, x] origin
    # top-left, top-right, bottom-right, bottom-left
    points_src: paddle.Tensor = paddle.to_tensor([[
        [start_x, start_y],
        [end_x, start_y],
        [end_x, end_y],
        [start_x, end_y],
    ]])

    # [y, x] destination
    # top-left, top-right, bottom-right, bottom-left
    points_dst: paddle.Tensor = paddle.to_tensor([[
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ]])  #.expand(points_src.shape[0], -1, -1)
    points_dst = paddle.expand(points_dst,shape=(points_src.shape[0], -1, -1))
    return crop_by_boxes(tensor,
                         points_src,
                         points_dst,
                         interpolation,
                         align_corners)


def crop_by_boxes(tensor: paddle.Tensor, src_box: paddle.Tensor, dst_box: paddle.Tensor,
                  interpolation: str = 'bilinear', align_corners: bool = False) -> paddle.Tensor:
    """Perform crop transform on 2D images (4D tensor) by bounding boxes.

    Given an input tensor, this function selected the interested areas by the provided bounding boxes (src_box).
    Then the selected areas would be fitted into the targeted bounding boxes (dst_box) by a perspective transformation.
    So far, the ragged tensor is not supported by PyTorch right now. This function hereby requires the bounding boxes
    in a batch must be rectangles with same width and height.

    Args:
        tensor (paddle.Tensor): the 2D image tensor with shape (B, C, H, W).
        src_box (paddle.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be extracted. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        dst_box (paddle.Tensor): a tensor with shape (B, 4, 2) containing the coordinates of the bounding boxes
            to be placed. The tensor must have the shape of Bx4x2, where each box is defined in the clockwise
            order: top-left, top-right, bottom-right and bottom-left. The coordinates must be in x, y order.
        interpolation (str): Interpolation flag. Default: 'bilinear'.
        align_corners (bool): mode for grid_generation. Default: False. See
          https://pypaddle.org/docs/stable/nn.functional.html#paddle.nn.functional.interpolate for details

    Returns:
        paddle.Tensor: the output tensor with patches.

    Examples:
        >>> input = paddle.arange(16, dtype=paddle.float32).reshape((1, 4, 4))
        >>> src_box = paddle.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ]])  # 1x4x2
        >>> dst_box = paddle.tensor([[
        ...     [0., 0.],
        ...     [1., 0.],
        ...     [1., 1.],
        ...     [0., 1.],
        ... ]])  # 1x4x2
        >>> crop_by_boxes(input, src_box, dst_box, align_corners=True)
        tensor([[[ 5.0000,  6.0000],
                 [ 9.0000, 10.0000]]])

    Note:
        If the src_box is smaller than dst_box, the following error will be thrown.
        RuntimeError: solve_cpu: For batch 0: U(2,2) is zero, singular U.
    """
    # validate_bboxes(src_box)
    # validate_bboxes(dst_box)

    assert len(tensor.shape) == 4, f"Only tensor with shape (B, C, H, W) supported. Got {tensor.shape}."

    # compute transformation between points and warp
    # Note: Tensor.dtype must be float. "solve_cpu" not implemented for 'Long'
    dst_trans_src: paddle.Tensor = get_perspective_transform(src_box, dst_box)
    # simulate broadcasting
    dst_trans_src = paddle.expand(dst_trans_src,shape=(tensor.shape[0], -1, -1))
    # dst_trans_src = dst_trans_src.expand(tensor.shape[0], -1, -1).type_as(tensor)

    bbox = infer_box_shape(dst_box)
    assert (bbox[0] == bbox[0][0]).all() and (bbox[1] == bbox[1][0]).all(), (
        f"Cropping height, width and depth must be exact same in a batch. Got height {bbox[0]} and width {bbox[1]}.")
    patches: paddle.Tensor = warp_affine(
        tensor, dst_trans_src[:, :2, :], (int(bbox[0][0].item()), int(bbox[1][0].item())),
        flags=interpolation, align_corners=align_corners)

    return patches

def warp_affine(src: paddle.Tensor, M: paddle.Tensor,
                dsize: Tuple[int, int], flags: str = 'bilinear',
                padding_mode: str = 'zeros',
                align_corners: bool = False) -> paddle.Tensor:
    r"""Applies an affine transformation to a tensor.

    The function warp_affine transforms the source tensor using
    the specified matrix:

    .. math::
        \text{dst}(x, y) = \text{src} \left( M_{11} x + M_{12} y + M_{13} ,
        M_{21} x + M_{22} y + M_{23} \right )

    Args:
        src (paddle.Tensor): input tensor of shape :math:`(B, C, H, W)`.
        M (paddle.Tensor): affine transformation of shape :math:`(B, 2, 3)`.
        dsize (Tuple[int, int]): size of the output image (height, width).
        mode (str): interpolation mode to calculate output values
          'bilinear' | 'nearest'. Default: 'bilinear'.
        padding_mode (str): padding mode for outside grid values
          'zeros' | 'border' | 'reflection'. Default: 'zeros'.
        align_corners (bool): mode for grid_generation. Default: False.

    Returns:
        paddle.Tensor: the warped tensor with shape :math:`(B, C, H, W)`.

    .. note::
       See a working example `here <https://kornia.readthedocs.io/en/latest/
       tutorials/warp_affine.html>`__.
    """
    if not isinstance(src, paddle.Tensor):
        raise TypeError("Input src type is not a paddle.Tensor. Got {}"
                        .format(type(src)))

    if not isinstance(M, paddle.Tensor):
        raise TypeError("Input M type is not a paddle.Tensor. Got {}"
                        .format(type(M)))

    if not len(src.shape) == 4:
        raise ValueError("Input src must be a BxCxHxW tensor. Got {}"
                         .format(src.shape))

    if not (len(M.shape) == 3 or M.shape[-2:] == (2, 3)):
        raise ValueError("Input M must be a Bx2x3 tensor. Got {}"
                         .format(M.shape))
    B, C, H, W = src.shape
    dsize_src = (H, W)
    out_size = dsize
    # we generate a 3x3 transformation matrix from 2x3 affine
    M_3x3: paddle.Tensor = convert_affinematrix_to_homography(M)
    dst_norm_trans_src_norm: paddle.Tensor = normalize_homography(
        M_3x3, dsize_src, out_size)
    src_norm_trans_dst_norm = paddle.inverse(dst_norm_trans_src_norm)
    grid = F.affine_grid(src_norm_trans_dst_norm[:, :2, :],
                         [B, C, out_size[0], out_size[1]],
                         align_corners=align_corners)
    return F.grid_sample(src, grid,
                         align_corners=align_corners,
                         mode=flags,
                         padding_mode=padding_mode)

def check_is_tensor(obj):
    """Checks whether the supplied object is a tensor.
    """
    if not isinstance(obj, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(type(obj)))
        
def normalize_homography(dst_pix_trans_src_pix: paddle.Tensor,
                         dsize_src: Tuple[int, int], dsize_dst: Tuple[int, int]) -> paddle.Tensor:
    r"""Normalize a given homography in pixels to [-1, 1].

    Args:
        dst_pix_trans_src_pix (paddle.Tensor): homography/ies from source to destiantion to be
          normalized. :math:`(B, 3, 3)`
        dsize_src (tuple): size of the source image (height, width).
        dsize_dst (tuple): size of the destination image (height, width).

    Returns:
        paddle.Tensor: the normalized homography of shape :math:`(B, 3, 3)`.
    """
    check_is_tensor(dst_pix_trans_src_pix)

    if not (len(dst_pix_trans_src_pix.shape) == 3 or dst_pix_trans_src_pix.shape[-2:] == (3, 3)):
        raise ValueError("Input dst_pix_trans_src_pix must be a Bx3x3 tensor. Got {}"
                         .format(dst_pix_trans_src_pix.shape))

    # source and destination sizes
    src_h, src_w = dsize_src
    dst_h, dst_w = dsize_dst

    # compute the transformation pixel/norm for src/dst
    src_norm_trans_src_pix: paddle.Tensor = normal_transform_pixel(
        src_h, src_w)
    src_pix_trans_src_norm = paddle.inverse(src_norm_trans_src_pix)
    dst_norm_trans_dst_pix: paddle.Tensor = normal_transform_pixel(
        dst_h, dst_w)

    # compute chain transformations
    dst_norm_trans_src_norm: paddle.Tensor = (
        dst_norm_trans_dst_pix @ (dst_pix_trans_src_pix @ src_pix_trans_src_norm)
    )
    return dst_norm_trans_src_norm

def normal_transform_pixel(height: int, width: int) -> paddle.Tensor:
    r"""Compute the normalization matrix from image size in pixels to [-1, 1].

    Args:
        height (int): image height.
        width (int): image width.

    Returns:
        paddle.Tensor: normalized transform with shape :math:`(1, 3, 3)`.
    """
    tr_mat = paddle.to_tensor([[1.0, 0.0, -1.0],
                           [0.0, 1.0, -1.0],
                           [0.0, 0.0, 1.0]])  # 3x3

    tr_mat[0, 0] = tr_mat[0, 0] * 2.0 / (width - 1.0)
    tr_mat[1, 1] = tr_mat[1, 1] * 2.0 / (height - 1.0)

    return tr_mat.unsqueeze(0)  # 1x3x3

def convert_affinematrix_to_homography(A: paddle.Tensor) -> paddle.Tensor:
    r"""Function that converts batch of affine matrices from [Bx2x3] to [Bx3x3].

    Examples::

        >>> input = paddle.rand(2, 2, 3)  # Bx2x3
        >>> output = kornia.convert_affinematrix_to_homography(input)  # Bx3x3
    """
    if not isinstance(A, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}".format(
            type(A)))
    if not (len(A.shape) == 3 and A.shape[-2:] == [2, 3]):
        raise ValueError("Input matrix must be a Bx2x3 tensor. Got {}"
                         .format(A.shape))
    return _convert_affinematrix_to_homography_impl(A)

def _convert_affinematrix_to_homography_impl(A: paddle.Tensor) -> paddle.Tensor:
    # A = np.pad(A.numpy(),[0, 0, 0, 1], "constant", value=0.)
    A = np.pad(A.numpy(),((0,0),(0,1),(0,0)), "constant")
    # H: paddle.Tensor = F.pad(A, [0, 0, 0, 1], "constant", value=0.,data_format='NCL')

    A[..., -1, -1] += 1.0
    H:paddle.Tensor = paddle.to_tensor(A)
    return H

def infer_box_shape(boxes: paddle.Tensor) -> Tuple[paddle.Tensor, paddle.Tensor]:
    r"""Auto-infer the output sizes for the given 2D bounding boxes.

    Args:
        boxes (paddle.Tensor): a tensor containing the coordinates of the
          bounding boxes to be extracted. The tensor must have the shape
          of Bx4x2, where each box is defined in the following (clockwise)
          order: top-left, top-right, bottom-right, bottom-left. The
          coordinates must be in the x, y order.

    Returns:
        Tuple[paddle.Tensor, paddle.Tensor]:
        - Bounding box heights, shape of :math:`(B,)`.
        - Boundingbox widths, shape of :math:`(B,)`.

    Example:
        >>> boxes = paddle.tensor([[
        ...     [1., 1.],
        ...     [2., 1.],
        ...     [2., 2.],
        ...     [1., 2.],
        ... ], [
        ...     [1., 1.],
        ...     [3., 1.],
        ...     [3., 2.],
        ...     [1., 2.],
        ... ]])  # 2x4x2
        >>> infer_box_shape(boxes)
        (tensor([2., 2.]), tensor([2., 3.]))
    """
    # validate_bboxes(boxes)
    width: paddle.Tensor = (boxes[:, 1, 0] - boxes[:, 0, 0] + 1)
    height: paddle.Tensor = (boxes[:, 2, 1] - boxes[:, 0, 1] + 1)
    return (height, width)

def get_perspective_transform(src, dst):
    r"""Calculates a perspective transform from four pairs of the corresponding
    points.

    The function calculates the matrix of a perspective transform so that:

    .. math ::

        \begin{bmatrix}
        t_{i}x_{i}^{'} \\
        t_{i}y_{i}^{'} \\
        t_{i} \\
        \end{bmatrix}
        =
        \textbf{map_matrix} \cdot
        \begin{bmatrix}
        x_{i} \\
        y_{i} \\
        1 \\
        \end{bmatrix}

    where

    .. math ::
        dst(i) = (x_{i}^{'},y_{i}^{'}), src(i) = (x_{i}, y_{i}), i = 0,1,2,3

    Args:
        src (Tensor): coordinates of quadrangle vertices in the source image.
        dst (Tensor): coordinates of the corresponding quadrangle vertices in
            the destination image.

    Returns:
        Tensor: the perspective transformation.

    Shape:
        - Input: :math:`(B, 4, 2)` and :math:`(B, 4, 2)`
        - Output: :math:`(B, 3, 3)`
    """
    if not isinstance(src, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}"
                        .format(type(src)))
    if not isinstance(dst, paddle.Tensor):
        raise TypeError("Input type is not a paddle.Tensor. Got {}"
                        .format(type(dst)))
    if not src.shape[-2:] == [4, 2]:
        raise ValueError("Inputs must be a Bx4x2 tensor. Got {}"
                         .format(src.shape))
    if not src.shape == dst.shape:
        raise ValueError("Inputs must have the same shape. Got {}"
                         .format(dst.shape))
    if not (src.shape[0] == dst.shape[0]):
        raise ValueError("Inputs must have same batch size dimension. Expect {} but got {}"
                         .format(src.shape, dst.shape))

    # we build matrix A by using only 4 point correspondence. The linear
    # system is solved with the least square method, so here
    # we could even pass more correspondence
    p = []
    for i in [0, 1, 2, 3]:
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'x'))
        p.append(_build_perspective_param(src[:, i], dst[:, i], 'y'))

    # A is Bx8x8
    A = paddle.stack(p, axis=1)

    # b is a Bx8x1
    b = paddle.stack([
        dst[:, 0:1, 0], dst[:, 0:1, 1],
        dst[:, 1:2, 0], dst[:, 1:2, 1],
        dst[:, 2:3, 0], dst[:, 2:3, 1],
        dst[:, 3:4, 0], dst[:, 3:4, 1],
    ], axis=1)

    # # solve the system Ax = b
    # X, LU = paddle.solve(b, A)
    X = np.linalg.solve(A.numpy(),b.numpy())
    # X = paddle.to_tensor(X)
    

    # create variable to return
    batch_size = src.shape[0]
    M = paddle.ones((batch_size, 9), dtype=src.dtype)
    M = np.ones((batch_size,9),dtype=np.float)
    M[..., :8] = np.squeeze(X, axis=-1)
    npm = M.reshape((-1, 3, 3))  # Bx3x3
    return paddle.to_tensor(npm)

def _build_perspective_param(p: paddle.Tensor, q: paddle.Tensor, axis: str) -> paddle.Tensor:
    ones = np.ones_like(p)[..., 0:1]
    # ones = paddle.ones_like(p)[..., 0:1]
    ones = paddle.to_tensor(ones)
    zeros = np.zeros_like(p)[..., 0:1]
    zeros = paddle.to_tensor(zeros)
    # zeros = paddle.zeros_like(p)[..., 0:1]
    if axis == 'x':
        return paddle.concat(
            [p[:, 0:1], p[:, 1:2], ones, zeros, zeros, zeros,
             -p[:, 0:1] * q[:, 0:1], -p[:, 1:2] * q[:, 0:1]
             ], axis=1)

    if axis == 'y':
        return paddle.concat(
            [zeros, zeros, zeros, p[:, 0:1], p[:, 1:2], ones,
             -p[:, 0:1] * q[:, 1:2], -p[:, 1:2] * q[:, 1:2]], axis=1)

    raise NotImplementedError(f"perspective params for axis `{axis}` is not implemented.")