import torch
import timm

class Bird1DBlock(torch.nn.Module):
    sampling_rate = 32000
    max_dilution = 7
    
    def __init__(self, conv_channels=64, kernel_size=5):
        super(Bird1DBlock, self).__init__()
        
        kernels = 16
        kernel_sizes = sorted(set(int(kernel_size * 2 ** (n/kernels)) for n in range(kernels)))
        self.kernel_size = kernel_size
        self.receptive_field = kernel_size * 4
        self.min_length = 2**self.max_dilution * kernel_size * 6

        self.input_conv = []
        for n in range(self.max_dilution):
            dilation = 2**n
            for ks in kernel_sizes:
                conv = torch.nn.Sequential(
                    torch.nn.Conv1d(in_channels=1, out_channels=conv_channels, kernel_size=ks, padding=0, stride=self.kernel_size*dilation, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=4, stride=2),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Conv1d(in_channels=conv_channels, out_channels=conv_channels, kernel_size=4, stride=2),
                    torch.nn.ReLU(inplace=True),
                )
                setattr(self, f'conv_{n}_{ks}', conv)
                self.input_conv += [conv, ]

        
    def forward(self, x):
        if x.shape[-1] < self.min_length:
            x = torch.nn.functional.pad(x, (0, self.min_length - x.shape[-1]))
        
        conv_out_size = x.shape[-1] // (self.receptive_field)
        input_conv_result = []
        for conv in self.input_conv:
            y = conv(x)
            y = torch.repeat_interleave(y, math.ceil(conv_out_size / y.shape[-1]), dim=-1)
            y = y[..., :conv_out_size]
            input_conv_result.append(y)

        input_conv_result = torch.stack(input_conv_result, dim=1)

        return torch.permute(input_conv_result, dims=[0, 2, 1, 3])


    
class Bird1DModel(torch.nn.Module):
    sampling_rate = 32000
    print = False
    
    def __init__(self, conv_channels=3, kernel_size=32, duration=5, step=5):
        super(Bird1DModel, self).__init__()
        
        self.duration = duration
        self.step = step
        self.kernel_size = kernel_size
        
        self.input_conv = Bird1DBlock(conv_channels=conv_channels, kernel_size=self.kernel_size)
        self.backbone = timm.create_model('resnet18d', pretrained=True, in_chans=conv_channels, num_classes=264)
        self.replace_bn_with_instance_norm()
    
        
    def replace_bn_with_instance_norm(self):
        resnet18 = self.backbone
        resnet18.norm_layers = []
        def get_inorm(layer):
            norm_layer = torch.nn.InstanceNorm2d(num_features = layer.num_features, 
                                                 eps=layer.eps, 
                                                 momentum=layer.momentum, 
                                                 affine=layer.affine, 
                                                 track_running_stats=layer.track_running_stats)
            norm_layer.load_state_dict(layer.state_dict())
            resnet18.norm_layers.append(norm_layer)
            return norm_layer

        resnet18.conv1[1] = get_inorm(resnet18.conv1[1])
        resnet18.conv1[4] = get_inorm(resnet18.conv1[4])
        resnet18.bn1 = get_inorm(resnet18.bn1)

        for ln in range(1, 5):
            for bn in range(2):
                getattr(resnet18, f'layer{ln}')[bn].bn1 = get_inorm(getattr(resnet18, f'layer{ln}')[bn].bn1)
                getattr(resnet18, f'layer{ln}')[bn].bn2 = get_inorm(getattr(resnet18, f'layer{ln}')[bn].bn2)

        for ln in range(2, 5):
            getattr(resnet18, f'layer{ln}')[0].downsample[2] = get_inorm(getattr(resnet18, f'layer{ln}')[0].downsample[2])    

        
    def forward(self, x):
        if self.print:
            print('Initial x:', x.shape)
            
        x = self.input_conv(x)
        if self.print:
            print('After conv:', x.shape)
        
        # Transform data so, that each interval correspond to individual item
        x = x.reshape((*x.shape[0:3], x.shape[-1] * self.input_conv.receptive_field // (self.sampling_rate * self.step), -1))
        x = torch.permute(x, (0, 3, 1, 2, 4))
        
        x = x.reshape((x.shape[0]*x.shape[1], *x.shape[2:]))

        if self.print:
            print('After split:', x.shape)
        
        # Process data in batches
        x = self.backbone(x)
        if self.print:
            print('After backbone:', x.shape)
        
        return x
