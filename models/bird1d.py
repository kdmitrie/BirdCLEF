import torch
import timm
from .adaptive_filter import AdaptiveFilter
from .modifier import replace_batchnorm2d_by_instancenorm2d


class Bird1DModel(torch.nn.Module):
    """
    1D model to process audio data

    Attributes
    ----------
    filter : torch.nn.Module
        the input filters, that transform a signal to 2d representation
    backbone : torch.nn.Module
        the feature extractor
    classifier : torch.nn.Module
        the final classifier
    """

    def __init__(self,
                 backbone: str = 'resnet18d',

                 in_channels: int = 1,
                 num_classes: int = 2,

                 filter_channels: int = 3,
                 filter_num_kernels: int = 16,
                 filter_max_dilation: int = 7,
                 filter_min_kernel_size: int = 32,

                 time_interval: int = 160000,
                 time_shift: int = 160000,

                 replace_bn: bool = True
                 ):
        """
        Parameters
        ----------
        backbone : str
            the name of timm model to be used as a feature extractor
        in_channels : int
            the number of input channels (default 1)
        num_classes : int
            the number of classes (default 2)

        filter_channels : int
            the number of input filter channels (default 3)
        filter_num_kernels : int
            the requested number of kernels;
            the actual number of kernels may be less, because the kernel size is kept unique
        filter_min_kernel_size : int
            the minimum size of a kernel; the maximum size is 2*min_kernel_size
        filter_max_dilation : int
            the number of dilated convolutional filters for each kernel size

        time_interval : int
            the interval to predict sound event (default 160000)

        time_shift : int
            the shift between the beginnings of two adjacent time intervals (default 160000)

        replace_bn: bool
            whether or not to replace batch norm layer in the backbone
        """

        super().__init__()

        self.filter = AdaptiveFilter(in_channels=in_channels,
                                     out_channels=filter_channels,
                                     num_kernels=filter_num_kernels,
                                     max_dilation=filter_max_dilation,
                                     min_kernel_size=filter_min_kernel_size)

        self.backbone = timm.create_model(backbone, pretrained=True, in_chans=filter_channels, num_classes=num_classes)
        self.classifier = self.backbone.get_classifier()
        self.backbone.reset_classifier(0)
        self._softmax = torch.nn.Softmax(dim=1)

        self._time_interval = time_interval // self.filter.get_receptive_field()
        self._time_shift = time_shift // self.filter.get_receptive_field()

        if replace_bn:
            replace_batchnorm2d_by_instancenorm2d(self.backbone)

    def __predict(self, x):
        if self.print:
            print('Initial x:', x.shape)

        x = self.input_conv(x)
        if self.print:
            print('After conv:', x.shape)

        # Transform data so, that each interval correspond to individual item
        x = x.reshape((*x.shape[0:3],
                       x.shape[-1] * self.input_conv.receptive_field // (self.sampling_rate * self.step), -1))
        x = torch.permute(x, (0, 3, 1, 2, 4))

        x = x.reshape((x.shape[0] * x.shape[1], *x.shape[2:]))

        if self.print:
            print('After split:', x.shape)

        # Process data in batches
        x = self.backbone(x)
        if self.print:
            print('After backbone:', x.shape)

        return x

    def forward(self, x):
        """Processing a signal (the only signal a time)"""
        # Apply adaptive filtering
        x = self.filter(x)

        # Split data and create a batch
        if x.shape[-1] > self._time_interval:
            data = [x[..., start:start + self._time_interval] for start in
                    range(0, x.shape[-1] - self._time_interval, self._time_shift)]
            x = torch.cat(data)
        else:
            x = torch.nn.functional.pad(x, (0, self._time_interval - x.shape[-1]))

        # Process data in batches
        x = self.backbone(x)
        x = torch.squeeze(x, dim=-1)

        x = self.classifier(x)
        x = self._softmax(x)

        # Calculate the probability of each class existence at least once
        x = 1 - torch.prod(1 - x, dim=0)

        return x[None, ...]
