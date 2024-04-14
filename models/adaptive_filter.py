import torch
import math


class AdaptiveFilter(torch.nn.Module):
    """
    Learnable wavelet-like filters to transform the 1d data into 2d data

    Attributes
    ----------
    in_channels : int
        the number of input channels (default 1)
    out_channels : int
        the number of output channels (default 3)

    num_kernels : int
        the requested number of kernels;
        the actual number of kernels may be less, because the kernel size is kept unique
    min_kernel_size : int
        the minimum size of a kernel; the maximum size is 2*min_kernel_size
    max_dilation : int
        the number of dilated convolutional filters for each kernel size 
    """

    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3,
                 num_kernels: int = 16,
                 max_dilation: int = 7,
                 min_kernel_size: int = 32
                 ):
        """
        Parameters
        ----------
        in_channels : int
            the number of input channels (default 1)
        out_channels : int
            the number of output channels (default 3)

        num_kernels : int
            the requested number of kernels;
            the actual number of kernels may be less, because the kernel size is kept unique
        min_kernel_size : int
            the minimum size of a kernel; the maximum size is 2*min_kernel_size
        max_dilation : int
            the number of dilated convolutional filters for each kernel size
        """

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.num_kernels = num_kernels
        self.min_kernel_size = min_kernel_size
        self.max_dilation = max_dilation

        # The list of actual kernel sizes
        kernels = sorted(set(int(self.min_kernel_size * 2 ** (n / num_kernels)) for n in range(num_kernels)))

        # The minimum length of input
        self.min_length = 2 ** self.max_dilation * self.min_kernel_size * 6

        # Below, the list of convolutional layers is created
        self.input_conv = torch.nn.ModuleList()
        for n in range(self.max_dilation):
            dilation = 2 ** n
            for ks in kernels:
                conv = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=ks,
                                    padding=0,
                                    stride=self.min_kernel_size * dilation, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4,
                                    stride=2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=4,
                                    stride=2),
                    torch.nn.ReLU(inplace=True),
                )
                self.input_conv.append(conv)

    def get_receptive_field(self):
        """The size of receptive field of the convolutional layer"""
        return self.min_kernel_size * 4

    def forward(self, x):
        # If data length is less than minimal, we pad it with zeros
        if x.shape[-1] < self.min_length:
            x = torch.nn.functional.pad(x, (0, self.min_length - x.shape[-1]))

        conv_out_size = x.shape[-1] // self.get_receptive_field()
        input_conv_result = []
        for conv in self.input_conv:
            y = conv(x)
            y = torch.repeat_interleave(y, math.ceil(conv_out_size / y.shape[-1]), dim=-1)
            y = y[..., :conv_out_size]
            input_conv_result.append(y)

        input_conv_result = torch.stack(input_conv_result, dim=1)

        return torch.permute(input_conv_result, dims=[0, 2, 1, 3])
