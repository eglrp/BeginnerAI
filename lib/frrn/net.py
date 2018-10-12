import torch
import numpy as np

class Conv2DGroupNormRelu(torch.nn.Module):
    def __init__(self, in_channels, n_filters, k_size, strides,padding,bias=True,dilation=1,n_groups=16):
        super(Conv2DGroupNormRelu, self).__init__()

        self.conv1 = torch.nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding,
                                     stride=strides, bias=bias, dilation=dilation)
        self.group1 = torch.nn.GroupNorm(n_groups, int(n_filters))

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.group1(outputs)
        outputs = torch.nn.ReLU(inplace=True)(outputs)

        return outputs

class Conv2DGroupNorm(torch.nn.Module):
    def __init__(self, in_channels, n_filters, k_size, strides,padding,bias=True,dilation=1,n_groups=16):
        super(Conv2DGroupNorm, self).__init__()

        self.conv1 = torch.nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size, padding=padding,
                                     stride=strides, bias=bias, dilation=dilation)
        self.group1 = torch.nn.GroupNorm(n_groups, int(n_filters))

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.group1(outputs)

        return outputs

class ResidualUnit(torch.nn.Module):
    def __init__(self, channels, kernel_size=3,strides=1,group_norm=False,n_groups=None):
        super(ResidualUnit, self).__init__()

        if group_norm:
            self.conv1 = Conv2DGroupNormRelu(channels, channels, k_size=kernel_size,strides=strides,padding=1,bias=False,
                                             n_groups=n_groups)
            self.conv2 = Conv2DGroupNorm(channels,channels, k_size=kernel_size,strides=strides,padding=1,bias=False,
                                         n_groups=n_groups)
        else:
            self.conv1 = Conv2DGroupNormRelu(channels, channels, k_size=kernel_size,strides=strides,padding=1,bias=False)
            self.conv2 = Conv2DGroupNorm(channels,channels, k_size=kernel_size,strides=strides,padding=1,bias=False)

    def forward(self, x):
        incoming = x
        x = self.conv1(x)
        x = self.conv2(x)

        return x + incoming

class FRRU(torch.nn.Module):
    """
    Full Resolution Residual Unit for FRRN
    """

    def __init__(self,prev_channels,out_channels,scale,group_norm=False,n_groups=None):
        super(FRRU, self).__init__()
        self.scale = scale
        self.prev_channels = prev_channels
        self.out_channels = out_channels
        self.group_norm = group_norm
        self.n_groups = n_groups


        if self.group_norm:
            conv_unit = Conv2DGroupNormRelu
            self.conv1 = conv_unit(
                prev_channels + 32, out_channels, k_size=3,
                stride=1, padding=1, bias=False, n_groups=self.n_groups
            )
            self.conv2 = conv_unit(
                out_channels, out_channels, k_size=3,
                stride=1, padding=1, bias=False, n_groups=self.n_groups
            )

        else:
            conv_unit = Conv2DGroupNormRelu
            self.conv1 = conv_unit(prev_channels + 32, out_channels, k_size=3,
                                   stride=1, padding=1, bias=False,)
            self.conv2 = conv_unit(out_channels, out_channels, k_size=3,
                                   stride=1, padding=1, bias=False,)

        self.conv_res = torch.nn.Conv2d(out_channels, 32, kernel_size=1, stride=1, padding=0)

    def forward(self, y, z):
        x = torch.cat([y, torch.nn.MaxPool2d(self.scale, self.scale)(z)], dim=1)
        y_prime = self.conv1(x)
        y_prime = self.conv2(y_prime)

        x = self.conv_res(y_prime)
        upsample_size = torch.Size([_s * self.scale for _s in y_prime.shape[-2:]])
        x = torch.nn.functional.upsample(x, size=upsample_size, mode="nearest")
        z_prime = z + x

        return y_prime, z_prime

class FRRN(torch.nn.Module):
    def __init__(self,n_classes=21,model_type=None,group_norm=False,n_groups=16):
        super(FRRN, self).__init__()
        self.n_classes = n_classes
        self.model_type = model_type
        self.group_norm = group_norm
        self.n_groups = n_groups

        if self.group_norm:
            self.conv1 = Conv2DGroupNormRelu(3, 48, 5, 1, 2)
        else:
            self.conv1 = Conv2DGroupNormRelu(3, 48, 5, 1, 2)

        self.up_residual_units = []
        self.down_residual_units = []
        for i in range(3):
            self.up_residual_units.append(ResidualUnit(channels=48,
                                             kernel_size=3,
                                             strides=1,
                                             group_norm=self.group_norm,
                                             n_groups=self.n_groups))
            self.down_residual_units.append(ResidualUnit(channels=48,
                                               kernel_size=3,
                                               strides=1,
                                               group_norm=self.group_norm,
                                               n_groups=self.n_groups))

        self.up_residual_units = torch.nn.ModuleList(self.up_residual_units)
        self.down_residual_units = torch.nn.ModuleList(self.down_residual_units)

        self.split_conv = torch.nn.Conv2d(
            48, 32, kernel_size=1, padding=0, stride=1, bias=False
        )

        # each spec is as (n_blocks, channels, scale)
        self.encoder_frru_specs = [[3, 96, 2], [4, 192, 4], [2, 384, 8], [2, 384, 16]]

        self.decoder_frru_specs = [[2, 192, 8], [2, 192, 4], [2, 48, 2]]

        # encoding
        prev_channels = 48
        self.encoding_frrus = {}
        for n_blocks, channels, scale in self.encoder_frru_specs:
            for block in range(n_blocks):
                key = "_".join(map(str, ["encoding_frru", n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                                        out_channels=channels,
                                        scale=scale,
                                        group_norm=self.group_norm,
                                        n_groups=self.n_groups),)
            prev_channels = channels

        # decoding
        self.decoding_frrus = {}
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(map(str, ["decoding_frru", n_blocks, channels, scale, block]))
                setattr(self, key, FRRU(prev_channels=prev_channels,
                                        out_channels=channels,
                                        scale=scale,
                                        group_norm=self.group_norm,
                                        n_groups=self.n_groups),)
            prev_channels = channels

        self.merge_conv = torch.nn.Conv2d(
            prev_channels + 32, 48, kernel_size=1, padding=0, stride=1, bias=False
        )

        self.classif_conv = torch.nn.Conv2d(
            48, self.n_classes, kernel_size=1, padding=0, stride=1, bias=True
        )

    def forward(self, x):
        x = self.conv1(x)
        # pass through residual units
        for i in range(3):
            x = self.up_residual_units[i](x)

        y = x
        z = self.split_conv(x)

        # encoding
        for n_blocks, channels, scale in self.encoder_frru_specs:
            # maxpool bigger feature map
            y_pooled = torch.nn.functional.max_pool2d(y, stride=2, kernel_size=2, padding=0)
            # pass through encoding FRRUs
            for block in range(n_blocks):
                key = "_".join(
                    map(str, ["encoding_frru", n_blocks, channels, scale, block])
                )
                y, z = getattr(self, key)(y_pooled, z)
            prev_channels = channels

        # decoding
        for n_blocks, channels, scale in self.decoder_frru_specs:
            # bilinear upsample smaller feature map
            upsample_size = torch.Size([_s * 2 for _s in y.size()[-2:]])
            y_upsampled = torch.nn.functional.upsample(y, size=upsample_size, mode="bilinear", align_corners=True)
            # pass through decoding FRRUs
            for block in range(n_blocks):
                key = "_".join(
                    map(str, ["decoding_frru", n_blocks, channels, scale, block])
                )
                # print("Incoming FRRU Size: ", key, y_upsampled.shape, z.shape)
                y, z = getattr(self, key)(y_upsampled, z)
                # print("Outgoing FRRU Size: ", key, y.shape, z.shape)
            prev_channels = channels

        # merge streams
        x = torch.cat([torch.nn.functional.upsample(y, scale_factor=2, mode="bilinear", align_corners=True), z], dim=1)
        x = self.merge_conv(x)

        # pass through residual units
        for i in range(3):
            x = self.down_residual_units[i](x)

        # final 1x1 conv to get classification
        x = self.classif_conv(x)

        return x