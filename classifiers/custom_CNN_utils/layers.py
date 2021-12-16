### Definitions of custom layers used in CNN
# Contains forward and backward pass implementations for following layers:
#   1. Affine
#   2. Convolutional ...
#   3. ReLU 
#   4. BatchNorm 
#   5. Dropout 
#
# The hyperparameters that will not be tuned are kept the same as those in the pytorch implementation
# These are implemented by extending Pytorch autograd Function as described here https://pytorch.org/docs/stable/notes/extending.html


## imports
import numpy as np
import torch
from torch import Tensor
from torch._C import device
from torch.nn.parameter import Parameter
from torch.nn import *
import math
from torch._torch_docs import reproducibility_notes
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from typing import Optional, List, Tuple, Union


## 1. Affine
class custom_AffineFunction(torch.autograd.Function):
    @staticmethod
    # forward: xw + b
    def forward(ctx, x, w, b):
        ctx.save_for_backward(x, w, b)

        # z = xw + b
        z = x.mm(w.t()) + b

        return z

    @staticmethod
    # backward: calculate gradients for x, w, and b according to regular formulas 
    def backward(ctx, dz):
        x, w, b = ctx.saved_tensors
        dx = dw = db = None

        # calculate gradients
        dx = dz.mm(w)
        dw = dz.t().mm(x)
        db = dz.sum(0)

        return dx, dw, db


## 2. Convolutional -- padding and different strides not used, but still implemented for completeness
# Implementation uses for loops so is slower than vectorized
class slow_custom_ConvFunction(torch.autograd.Function):
    @staticmethod
    # forward: 
    def forward(ctx, x, w, b, stride, pad, dilation, groups):
        # get shapes of input and filters
        M, C, x_H, x_W = x.shape  
        F, _, w_H, w_W = w.shape  # F: number of filters

        # determine shape of output according to formula, and init output
        z_H = 1 + (x_H + 2 * pad - w_H) // stride
        z_W = 1 + (x_W + 2 * pad - w_W) // stride
        z = torch.zeros((M, F, z_H, z_W))

        # pad input with 0s -- need to send x to cpu first, then back to cuda if using GPU
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0)

        # loops
        for m in range(M): # over data 
            for f in range(F): # over filter
                for j in range(z_H): # over height of z
                    for i in range(z_W): # over width of z
                        # perform convolution
                        z[m, f, j, i] = torch.sum(x_padded[m, 
                                                        :, 
                                                        j*stride : j*stride + w_H, 
                                                        i*stride : i*stride + w_W] 
                                                * w[f, :, :, :]) + b[f]

        # store for backward pass
        ctx.save_for_backward(x, w, b)
        ctx.stride = stride
        ctx.pad = pad

        # return
        return z 

    @staticmethod
    # backward: 
    def backward(ctx, dz):
        # extract from context ctx
        x, w, b = ctx.saved_tensors  # need comma since this returns a tuple
        dx, dw, db, dstride, dpad, ddilation, dgroups = None, None, None, None, None, None, None
        stride = ctx.stride 
        pad = ctx.pad

        # get shapes of input and filters
        M, C, x_H, x_W = x.shape  
        F, _, w_H, w_W = w.shape  # F: number of filters

        # Padding (same as forward)
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0)

        # determine shape of output according to formula
        z_H = 1 + (x_H + 2 * pad - w_H) // stride
        z_W = 1 + (x_W + 2 * pad - w_W) // stride

        # init gradients
        dx_padded = torch.zeros(x_padded.shape)
        dw = torch.zeros(w.shape)
        db = torch.zeros(b.shape)

        # loops
        for m in range(M):
            for f in range(F):
                # db = sum across filters axis
                db[f] += dz[m, f].sum()  

                for j in range(z_H):
                    for i in range(z_W):
                        # dw = x * dz
                        dw[f] += x_padded[m, 
                                          :, 
                                          j*stride : j*stride + w_H, 
                                          i*stride : i*stride + w_W] * dz[m, f, j, i]

                        
                        # dx = w * dz 
                        dx_padded[m, 
                                  :, 
                                  j*stride : j*stride + w_H, 
                                  i*stride : i*stride + w_W] += w[f] * dz[m, f, j, i]

        # Extract dx from dx_pad (H and W dimensions were padded)
        dx = dx_padded[:, :, pad: pad+x_H, pad : pad+x_W]

        # return (as many as input to fwd pass)
        return dx, dw, db, dstride, dpad, ddilation, dgroups



class custom_ConvFunction(torch.autograd.Function):
    @staticmethod
    # forward: 
    def forward(ctx, x, w, b, stride, pad, dilation, groups):
        # get shapes of input and filters
        M, C, x_H, x_W = x.shape  
        F, _, w_H, w_W = w.shape  # F: number of filters

        # pad input with 0s -- need to send x to cpu first, then back to cuda if using GPU
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0).to('cuda:0')

        # determine shape of output according to formula, and init output
        z_H = 1 + (x_H + 2 * pad - w_H) // stride
        z_W = 1 + (x_W + 2 * pad - w_W) // stride
        z = torch.zeros((M, F, z_H, z_W))

        # loops
        for m in range(M): # over data 
            for f in range(F): # over filter
                for j in range(z_H): # over height of z
                    for i in range(z_W): # over width of z
                        # perform convolution
                        z[m, f, j, i] = torch.sum(x_padded[m, 
                                                        :, 
                                                        j*stride : j*stride + w_H, 
                                                        i*stride : i*stride + w_W] 
                                                * w[f, :, :, :]) + b[f]

        # store for backward pass
        ctx.save_for_backward(x, w, b)
        ctx.stride = stride
        ctx.pad = pad

        # return
        return z 

    @staticmethod
    # backward: 
    def backward(ctx, dz):
        # extract from context ctx
        x, w, b = ctx.saved_tensors  # need comma since this returns a tuple
        dx, dw, db, dstride, dpad, ddilation, dgroups = None, None, None, None, None, None, None
        stride = ctx.stride 
        pad = ctx.pad

        # get shapes of input and filters
        M, C, x_H, x_W = x.shape  
        F, _, w_H, w_W = w.shape  # F: number of filters

        # Padding (same as forward)
        x_padded = torch.nn.functional.pad(x, (pad,pad,pad,pad), 'constant', 0)

        # determine shape of output according to formula
        z_H = 1 + (x_H + 2 * pad - w_H) // stride
        z_W = 1 + (x_W + 2 * pad - w_W) // stride

        # init gradients
        dx_padded = torch.zeros(x_padded.shape)
        dw = torch.zeros(w.shape)
        db = torch.zeros(b.shape)

        # loops
        for m in range(M):
            for f in range(F):
                # db = sum across filters axis
                db[f] += dz[m, f].sum()  

                for j in range(z_H):
                    for i in range(z_W):
                        # dw = x * dz
                        dw[f] += x_padded[m, 
                                          :, 
                                          j*stride : j*stride + w_H, 
                                          i*stride : i*stride + w_W] * dz[m, f, j, i]

                        
                        # dx = w * dz 
                        dx_padded[m, 
                                  :, 
                                  j*stride : j*stride + w_H, 
                                  i*stride : i*stride + w_W] += w[f] * dz[m, f, j, i]

        # Extract dx from dx_pad (H and W dimensions were padded)
        dx = dx_padded[:, :, pad: pad+x_H, pad : pad+x_W]

        # return (as many as input to fwd pass)
        return dx, dw, db, dstride, dpad, ddilation, dgroups


# 3. ReLU
class custom_ReLUFunction(torch.autograd.Function):
    @staticmethod
    # forward: threshold < 0 values to 0
    def forward(ctx, x):
        ctx.save_for_backward(x)

        # need to make copy first, then threshold
        z = x.clone()
        z[x < 0] = 0

        return z

    @staticmethod
    # backward: dx = 0 if x < 0, dz otherwise
    def backward(ctx, dz):
        x, = ctx.saved_tensors  # need comma since this returns a tuple
        dx = None

        # calculate gradient 
        dx = dz.clone()
        dx[x < 0] = 0

        return dx


# 4. batchnorm 
class custom_Batchnorm2DFunction(torch.autograd.Function):
    @staticmethod
    # forward: 
    def forward(ctx, x, r_mean, r_var, gamma, beta, train, momentum, eps):       
        # need to send new tensors to appropriate device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # get shape first and then reshape to have C last (and also flatten)
        M, C, H, W = x.shape
        x_flat = x.permute(0,2,3,1).reshape(M*H*W, C).to(device)

        # if training
        if train:

            # get mean and std
            mu = x_flat.mean(axis=0)
            var = x_flat.var(axis=0)
            std = torch.sqrt(var).to(device)

            # normalize and then scale and shift using gamma and beta (need to broadcast)
            a = ((x_flat - mu)/std).to(device)
            z = gamma * a + beta

            # update running mean and vars
            r_mean = momentum * r_mean + (1 - momentum) * mu
            r_var = momentum * r_var + (1 - momentum) * (std**2)

            # store required variables for backprop 
            ctx.save_for_backward(std, gamma, a)

        # for testing just normalize using running means and var, and then scale and shift
        else:
            # normalize, then scale and shift
            a = ((x_flat - r_mean) / torch.sqrt(r_var + eps)).to(device)
            z = gamma * a + beta

        # reshape z back to same shape as x
        z = z.reshape(M, H, W, C).permute(0,3,1,2) # reshape back
        return z

    @staticmethod
    # backward: 
    def backward(ctx, dz):
        # extract from context
        std, gamma, a = ctx.saved_tensors  # need comma since this returns a tuple
        dx, dr_mean, dr_var, dgamma, dbeta, dtrain, dmomentum, deps = None, None, None, None, None, None, None, None

        # reshape dz
        M, C, H, W = dz.shape
        dz = dz.permute(0,2,3,1).reshape(M*H*W, C)

        ## calculate gradients accoridng to formulation (reverse order of above computations)
        # beta and gamma 
        dbeta = dz.sum(axis=0)
        dgamma = (dz * a).sum(axis=0)

        # x
        df_da = dz * gamma
        sum_df_da = df_da.sum(axis=0)
        dx = (df_da - ((df_da * a).sum(axis=0) * a/M) - sum_df_da/M) / std

        # reshape dx
        dx = dx.reshape(M, H, W, C).permute(0,3,1,2)

        # return (return all 6 since we stored that many, but some remain None from the start) 
        # --> note order matters and is same as saving
        return dx, dr_mean, dr_var, dgamma, dbeta, dtrain, dmomentum, deps



# 5. dropout
class custom_DropoutFunction(torch.autograd.Function):
    @staticmethod
    # forward: if train is True, randomly drop weights with prob p 
    def forward(ctx, x, p, train): 
        ctx.save_for_backward(x)

        # need to send new tensors to appropriate device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if train is True, randomly drop weights with prob p 
        mask = None
        if train:
            # use a scaled boolean mask calculated randomly according to p
            mask = (torch.rand(x.shape, device=device) < p) / p
            z = mask.to(device) * x
        
        # for testing, it basically is the identity function
        else:   
            z = x

        # save variables
        ctx.mask = mask
        ctx.train = train

        return z

    @staticmethod
    # backward: 
    def backward(ctx, dz):
        # extract from context
        x, = ctx.saved_tensors  # need comma since this returns a tuple
        mask = ctx.mask
        train = ctx.train
        dx = None

        # use the same mask for train
        if train:
            dx = dz * mask
        
        # for test, derivative is just same as dz 
        else:
            dx = dz

        # return (two Nones since we gave p and train as inputs)
        return dx, None, None










# --------------------------
# nn.Module Wrappers to use the autograd functions (defined above) 1-5 inside nn.Sequential
# These are taken from the PyTorch documentation, and only the autograd functions are replaced
# This replacement only requires a change in the forward() function of each Module,  
#    e.g. from F.linear(...) to AffineFunction.apply(...)
# And also a change in the super() call since the name of the class is changed

## 1. Affine 
class custom_Linear(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(custom_Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        return custom_AffineFunction.apply(input, self.weight, self.bias) # CHANGED HERE

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


## 2. Conv 
class custom_Conv2d(torch.nn.modules.conv._ConvNd):
    __doc__ = r"""Applies a 2D convolution over an input signal composed of several input
    planes.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{\text{in}}, H, W)` and output :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j}) = \text{bias}(C_{\text{out}_j}) +
        \sum_{k = 0}^{C_{\text{in}} - 1} \text{weight}(C_{\text{out}_j}, k) \star \text{input}(N_i, k)


    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    """ + r"""

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    * :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.

    * :attr:`padding` controls the amount of padding applied to the input. It
      can be either a string {{'valid', 'same'}} or a tuple of ints giving the
      amount of implicit padding applied on both sides.

    * :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.

    {groups_note}

    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:

        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension

    Note:
        {depthwise_separable_note}

    Note:
        {cudnn_reproducibility_note}

    Note:
        ``padding='valid'`` is the same as no padding. ``padding='same'`` pads
        the input so the output has the shape as the input. However, this mode
        doesn't support any stride values other than 1.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int, tuple or str, optional): Padding added to all four sides of
            the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``,
            ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input
            channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``
    """ + r"""

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where

          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor

          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor

    Attributes:
        weight (Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
            :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
            The values of these weights are sampled from
            :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape
            (out_channels). If :attr:`bias` is ``True``,
            then the values of these weights are
            sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`

    Examples:

        >>> # With square kernels and equal stride
        >>> m = nn.Conv2d(16, 33, 3, stride=2)
        >>> # non-square kernels and unequal stride and with padding
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
        >>> # non-square kernels and unequal stride and with padding and dilation
        >>> m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
        >>> input = torch.randn(20, 16, 50, 100)
        >>> output = m(input)

    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation

    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(custom_Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        #if self.padding_mode != 'zeros':
        #    return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
        #                    weight, bias, self.stride,
        #                    _pair(0), self.dilation, self.groups)
        return custom_ConvFunction.apply(input, weight, bias, self.stride[0],
                        self.padding[0], self.dilation, self.groups)

    def forward(self, input: Tensor) -> Tensor:
        return self._conv_forward(input, self.weight, self.bias)


## 3. ReLU
class custom_ReLU(Module):
    r"""Applies the rectified linear unit function element-wise:

    :math:`\text{ReLU}(x) = (x)^+ = \max(0, x)`

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Output: :math:`(*)`, same shape as the input.

    .. image:: ../scripts/activation_images/ReLU.png

    Examples::

        >>> m = nn.ReLU()
        >>> input = torch.randn(2)
        >>> output = m(input)


      An implementation of CReLU - https://arxiv.org/abs/1603.05201

        >>> m = nn.ReLU()
        >>> input = torch.randn(2).unsqueeze(0)
        >>> output = torch.cat((m(input),m(-input)))
    """
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(custom_ReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return custom_ReLUFunction.apply(input)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


## 4. BatchNorm -- 2 classes -- changes made to only first one _BatchNorm
class _BatchNorm(torch.nn.modules.batchnorm._NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        # find device:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_BatchNorm, self).__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return custom_Batchnorm2DFunction.apply(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class custom_BatchNorm2d(_BatchNorm):
    r"""Applies Batch Normalization over a 4D input (a mini-batch of 2D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .

    .. math::

        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The mean and standard-deviation are calculated per-dimension over
    the mini-batches and :math:`\gamma` and :math:`\beta` are learnable parameter vectors
    of size `C` (where `C` is the input size). By default, the elements of :math:`\gamma` are set
    to 1 and the elements of :math:`\beta` are set to 0. The standard-deviation is calculated
    via the biased estimator, equivalent to `torch.var(input, unbiased=False)`.

    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.

    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.

    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.

    Because the Batch Normalization is done over the `C` dimension, computing statistics
    on `(N, H, W)` slices, it's common terminology to call this Spatial Batch Normalization.

    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, H, W)`
        eps: a value added to the denominator for numerical stability.
            Default: 1e-5
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``

    Shape:
        - Input: :math:`(N, C, H, W)`
        - Output: :math:`(N, C, H, W)` (same shape as input)

    Examples::

        >>> # With Learnable Parameters
        >>> m = nn.BatchNorm2d(100)
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm2d(100, affine=False)
        >>> input = torch.randn(20, 100, 35, 45)
        >>> output = m(input)
    """

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


## 5. Dropout

class custom_Dropout(torch.nn.modules.dropout._DropoutNd):
    r"""During training, randomly zeroes some of the elements of the input
    tensor with probability :attr:`p` using samples from a Bernoulli
    distribution. Each channel will be zeroed out independently on every forward
    call.

    This has proven to be an effective technique for regularization and
    preventing the co-adaptation of neurons as described in the paper
    `Improving neural networks by preventing co-adaptation of feature
    detectors`_ .

    Furthermore, the outputs are scaled by a factor of :math:`\frac{1}{1-p}` during
    training. This means that during evaluation the module simply computes an
    identity function.

    Args:
        p: probability of an element to be zeroed. Default: 0.5
        inplace: If set to ``True``, will do this operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(*)`. Input can be of any shape
        - Output: :math:`(*)`. Output is of the same shape as input

    Examples::

        >>> m = nn.Dropout(p=0.2)
        >>> input = torch.randn(20, 16)
        >>> output = m(input)

    .. _Improving neural networks by preventing co-adaptation of feature
        detectors: https://arxiv.org/abs/1207.0580
    """

    def forward(self, input: Tensor) -> Tensor:
        return custom_DropoutFunction.apply(input, self.p, self.training)

