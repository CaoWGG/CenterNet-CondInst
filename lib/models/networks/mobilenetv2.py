from torch import nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import math
import torch

__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self,width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1], # 0
            [6, 24, 2, 2], # 1
            [6, 32, 3, 2], # 2
            [6, 64, 4, 2], # 3
            [6, 96, 3, 1], # 4
            [6, 160, 3, 2],# 5
            [6, 320, 1, 1],# 6
        ]
        self.feat_id = [0,1,2,4,6]
        self.feat_channel = []
        # building first layer
        input_channel = int(input_channel * width_mult)
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for id,(t, c, n, s) in enumerate(inverted_residual_setting):
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
            if id in self.feat_id  :
                self.__setattr__("feature_%d"%id,nn.Sequential(*features))
                self.feat_channel.append(output_channel)
                features = []

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        y = []
        for id in self.feat_id:
            x = self.__getattr__("feature_%d"%id)(x)
            y.append(x)
        return y

def load_model(model,state_dict):
    new_model=model.state_dict()
    new_keys = list(new_model.keys())
    old_keys = list(state_dict.keys())
    restore_dict = OrderedDict()
    for id in range(len(new_keys)):
        restore_dict[new_keys[id]] = state_dict[old_keys[id]]
    model.load_state_dict(restore_dict)


def mobilenetv2_10(pretrained=True, **kwargs):
    model = MobileNetV2(width_mult = 1.0)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['mobilenet_v2'],
                                              progress=True)
        load_model(model,state_dict)
    return model


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]

def fill_fc_weights(layers):
    for m in layers.modules():
        if isinstance(m, nn.Conv2d):
            #nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

class MobileNetFuse(nn.Module):
    def __init__(self,chanels):
        super(MobileNetFuse, self).__init__()
        self.small = nn.Sequential(ConvBNReLU(in_planes=chanels[0],out_planes=chanels[1],kernel_size=1,stride=1),
                                   nn.Upsample(scale_factor=2,mode='nearest'))
        self.mid = ConvBNReLU(in_planes=chanels[1],out_planes=chanels[1],kernel_size=1,stride=1)
        self.large = ConvBNReLU(in_planes=chanels[2],out_planes=chanels[1],kernel_size=3,stride=2)

        self.att_small = ConvBNReLU(in_planes=chanels[1], out_planes=8, kernel_size=1, stride=1)
        self.att_mid = ConvBNReLU(in_planes=chanels[1], out_planes=8, kernel_size=1, stride=1)
        self.att_lagre = ConvBNReLU(in_planes=chanels[1], out_planes=8, kernel_size=1, stride=1)

        self.attention = nn.Conv2d(8*3,3,kernel_size=1,stride=1,padding=0,bias=True)

    def forward(self, x):
        small,mid,large = x
        small = self.small(small)
        mid = self.mid(mid)
        large = self.large(large)

        att_small = self.att_small(small)
        att_mid = self.att_small(mid)
        att_large = self.att_small(large)

        att_feat = torch.cat([att_small,att_mid,att_large],1)
        attention = self.attention(att_feat)
        attention = 2 * F.softmax(attention,dim=1)
        att_small,att_mid,att_large = torch.split(attention,[1,1,1],dim=1)
        fuse = small*att_small + mid*att_mid + large*att_large
        return fuse


class MobileNetFPN(nn.Module):
    def __init__(self,channels):
        super(MobileNetFPN, self).__init__()
        channels = channels[::-1]
        self.level_0 = MobileNetFuse(channels[:3])
        self.level_1 = MobileNetFuse(channels[1:4])
        self.level_2 = MobileNetFuse(channels[2:5])

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x[::-1]
        x[1] = self.level_0(x[:3])
        x[2] = self.level_1(x[1:4])
        x = self.level_2(x[2:5])
        return x

class MobileNetSeg(nn.Module):
    def __init__(self, base_name,heads,head_conv=24, pretrained = True):
        super(MobileNetSeg, self).__init__()
        self.heads = heads
        self.base = globals()[base_name](
            pretrained=pretrained)
        channels = self.base.feat_channel
        self.fpn = MobileNetFPN(channels)
        for head in self.heads:
            classes = self.heads[head]
            if head == 'seg_feat':
                fc = nn.Sequential(
                    nn.Conv2d(channels[self.first_level], classes,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True)
                )
            else:
                fc = nn.Sequential(
                    nn.Conv2d(channels[1], head_conv,
                              kernel_size=3, padding=1, bias=True),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(head_conv, classes,
                              kernel_size=1, stride=1,
                              padding=0, bias=True))
            fill_fc_weights(fc)
            if 'hm' in head:
                fc[-1].bias.data.fill_(-9.2103)



            self.__setattr__(head, fc)

    def forward(self, x):
        x = self.base(x)
        x = self.fpn(x)
        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        return [ret]



def get_mobile_net(num_layers, heads, head_conv=24):
  model = MobileNetSeg('mobilenetv2_{}'.format(num_layers), heads,
                 pretrained=True,
                 head_conv=head_conv)
  return model

if __name__ == '__main__':
    model = get_mobile_net(10,{'hm':2,'reg':2,'wh':2},64)
    input=torch.zeros([1,3,416,416])
    model(input)
