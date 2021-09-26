import paddle
from paddle import nn
from paddle.nn import functional as F


def normalize_kernel2d(input: paddle.Tensor) -> paddle.Tensor:
    r"""Normalizes both derivative and smoothing kernel.
    """
    if len(input.shape) < 2:
        raise TypeError("input should be at least 2D to_tensor. Got {}"
                        .format(input.shape))
    norm: paddle.Tensor = input.abs().sum(axis=-1).sum(axis=-1)
    return input / (norm.unsqueeze(-1).unsqueeze(-1))


def get_sobel_kernel_3x3() -> paddle.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return paddle.to_tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.],
    ])


def get_sobel_kernel_5x5_2nd_order() -> paddle.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return paddle.to_tensor([
        [-1., 0., 2., 0., -1.],
        [-4., 0., 8., 0., -4.],
        [-6., 0., 12., 0., -6.],
        [-4., 0., 8., 0., -4.],
        [-1., 0., 2., 0., -1.]
    ])


def _get_sobel_kernel_5x5_2nd_order_xy() -> paddle.Tensor:
    """Utility function that returns a 2nd order sobel kernel of 5x5"""
    return paddle.to_tensor([
        [-1., -2., 0., 2., 1.],
        [-2., -4., 0., 4., 2.],
        [0., 0., 0., 0., 0.],
        [2., 4., 0., -4., -2.],
        [1., 2., 0., -2., -1.]
    ])


def get_diff_kernel_3x3() -> paddle.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return paddle.to_tensor([
        [-0., 0., 0.],
        [-1., 0., 1.],
        [-0., 0., 0.],
    ])


def get_diff_kernel3d() -> paddle.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: paddle.Tensor = paddle.to_tensor([[[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [-0.5, 0.0, 0.5],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
                                [[[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],

                                [[0.0, -0.5, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.0]],

                                [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],
                                ],
                                [[[0.0, 0.0, 0.0],
                                [0.0, -0.5, 0.0],
                                [0.0, 0.0, 0.0]],

                                [[0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0]],

                                [[0.0, 0.0, 0.0],
                                [0.0, 0.5, 0.0],
                                [0.0, 0.0, 0.0]],
                                ],
                            ])
    return kernel.unsqueeze(1)


def get_diff_kernel3d_2nd_order() -> paddle.Tensor:
    """Utility function that returns a first order derivative kernel of 3x3x3"""
    kernel: paddle.Tensor = paddle.to_tensor([[[[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [1.0, -2.0, 1.0],
                                           [0.0, 0.0, 0.0]],

                                          [[0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0],
                                           [0.0, 0.0, 0.0]],
                                          ],
        [[[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[0.0, 1.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 1.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         ],
        [[[0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [0.0, -2.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0]],
         ],
        [[[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[1.0, 0.0, -1.0],
          [0.0, 0.0, 0.0],
          [-1.0, 0.0, 1.0]],

         [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],
         ],
        [[[0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, -1.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[0.0, -1.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0]],
         ],
        [[[0.0, 0.0, 0.0],
          [1.0, 0.0, -1.0],
          [0.0, 0.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0],
          [0.0, 0.0, 0.0]],

         [[0.0, 0.0, 0.0],
          [-1.0, 0.0, 1.0],
          [0.0, 0.0, 0.0]],
         ],
    ])
    return kernel.unsqueeze(1)


def get_sobel_kernel2d() -> paddle.Tensor:
    kernel_x: paddle.Tensor = get_sobel_kernel_3x3()
    kernel_y: paddle.Tensor = kernel_x.transpose(0, 1)
    return paddle.stack([kernel_x, kernel_y])


def get_diff_kernel2d() -> paddle.Tensor:
    kernel_x: paddle.Tensor = get_diff_kernel_3x3()
    kernel_y: paddle.Tensor = kernel_x.transpose(0, 1)
    return paddle.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order() -> paddle.Tensor:
    gxx: paddle.Tensor = get_sobel_kernel_5x5_2nd_order()
    gyy: paddle.Tensor = gxx.transpose(0, 1)
    gxy: paddle.Tensor = _get_sobel_kernel_5x5_2nd_order_xy()
    return paddle.stack([gxx, gxy, gyy])


def get_diff_kernel2d_2nd_order() -> paddle.Tensor:
    gxx: paddle.Tensor = paddle.to_tensor([
        [0., 0., 0.],
        [1., -2., 1.],
        [0., 0., 0.],
    ])
    gyy: paddle.Tensor = gxx.transpose(0, 1)
    gxy: paddle.Tensor = paddle.to_tensor([
        [-1., 0., 1.],
        [0., 0., 0.],
        [1., 0., -1.],
    ])
    return paddle.stack([gxx, gxy, gyy])


def get_sobel_kernel2d() -> paddle.Tensor:
    kernel_x: paddle.Tensor = get_sobel_kernel_3x3()
    kernel_y: paddle.Tensor = kernel_x.transpose([1,0])
    return paddle.stack([kernel_x, kernel_y])


def get_spatial_gradient_kernel2d(mode: str, order: int) -> paddle.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff']:
        raise TypeError("mode should be either sobel\
                         or diff. Got {}".format(mode))
    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))
    if mode == 'sobel' and order == 1:
        kernel: paddle.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    else:
        raise NotImplementedError("")
    return kernel


class SpatialGradient(nn.Layer):
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    Return:
        paddle.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        >>> input = paddle.rand(1, 3, 4, 4)
        >>> output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode: str = 'sobel',
                 order: int = 1,
                 normalized: bool = True) -> None:
        super(SpatialGradient, self).__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode
        self.kernel = get_spatial_gradient_kernel2d(mode, order)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'normalized=' + str(self.normalized) + ', ' + \
            'mode=' + self.mode + ')'

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        if not paddle.is_tensor(input):
            raise TypeError("Input type is not a paddle.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # prepare kernel
        b, c, h, w = input.shape
        tmp_kernel: paddle.Tensor = self.kernel.detach()
        kernel: paddle.Tensor = tmp_kernel.unsqueeze(1).unsqueeze(1)

        # convolve input to_tensor with sobel kernel
        kernel_flip: paddle.Tensor = kernel.flip([-3])
        # Pad with "replicate for spatial dims, but with zeros for channel
        spatial_pad = [self.kernel.shape[1] // 2,
                       self.kernel.shape[1] // 2,
                       self.kernel.shape[2] // 2,
                       self.kernel.shape[2] // 2]
        out_channels: int = 3 if self.order == 2 else 2
        fout = F.pad(input.reshape(
            [b * c, 1, h, w]), spatial_pad, 'replicate').numpy()
        fout = fout[:,:,None]
        # padded_inp: paddle.Tensor = F.pad(input.reshape(
        #     [b * c, 1, h, w]), spatial_pad, 'replicate')[:, :, None]
        padded_inp: paddle.Tensor = paddle.to_tensor(fout)
        return F.conv3d(padded_inp, kernel_flip, padding=0).reshape((b, c, out_channels, h, w))


def spatial_gradient(input: paddle.Tensor,
                     mode: str = 'sobel',
                     order: int = 1,
                     normalized: bool = True) -> paddle.Tensor:
    r"""Computes the first order image derivative in both x and y using a Sobel
    operator.

    See :class:`~kornia.filters.SpatialGradient` for details.
    """
    return SpatialGradient(mode, order, normalized)(input)


class Sobel(nn.Layer):
    r"""Computes the Sobel operator and returns the magnitude per channel.

    Return:
        paddle.Tensor: the sobel edge gradient maginitudes map.

    Args:
        normalized (bool): if True, L1 norm of the kernel is set to 1.
        eps (float): regularization number to avoid NaN during backprop. Default: 1e-6.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples:
        >>> input = paddle.rand(1, 3, 4, 4)
        >>> output = kornia.filters.Sobel()(input)  # 1x3x4x4
    """

    def __init__(self,
                 normalized: bool = True, eps: float = 1e-6) -> None:
        super(Sobel, self).__init__()
        self.normalized: bool = normalized
        self.eps: float = eps

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'normalized=' + str(self.normalized) + ')'

    def forward(self, input: paddle.Tensor) -> paddle.Tensor:  # type: ignore
        if not paddle.is_tensor(input):
            raise TypeError("Input type is not a paddle.Tensor. Got {}"
                            .format(type(input)))
        if not len(input.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(input.shape))
        # comput the x/y gradients
        edges: paddle.Tensor = spatial_gradient(input,
                                                normalized=self.normalized)

        # unpack the edges
        gx: paddle.Tensor = edges[:, :, 0]
        gy: paddle.Tensor = edges[:, :, 1]

        # compute gradient maginitude
        magnitude: paddle.Tensor = paddle.sqrt(gx * gx + gy * gy + self.eps)
        return magnitude


def sobel(inputs: paddle.Tensor, normalized: bool = True, eps: float = 1e-6):
    return Sobel(normalized,eps)(inputs)
