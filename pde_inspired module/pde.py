import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
import numpy as np
from .common import BaseNetwork
from .FD import FD3d


class L1:
    def __init__(
        self,
    ):
        self.calc = torch.nn.L1Loss()

    def __call__(self, x, y):
        return self.calc(x, y)


class Generator(BaseNetwork):
    def __init__(self):  # 1046
        super(Generator, self).__init__()

        self.c_generator = CoarseGenerator()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(4, 64, 7),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(True),
        )

        rates = '1+2+4+8'
        rates = list(map(int, list(rates.split("+"))))
        self.middle = nn.Sequential(*[AOTBlock(256, rates) for _ in range(4)])

        self.decoder = nn.Sequential(
            UpConv(256, 128), nn.ReLU(True), UpConv(128, 64), nn.ReLU(True), nn.Conv2d(64, 3, 3, stride=1, padding=1)
        )

        self.init_weights()
        self.l1 = L1()

    def forward(self, x, mask):
        x0 = self.c_generator(x)
        c_loss = self.l1(x, x0)
        x1 = x0 * mask + x * (1. - mask)
        x = torch.cat([x1, mask], dim=1)
        # x = torch.cat([x, x1], dim=1)

        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        return x, c_loss


class UpConv(nn.Module):
    def __init__(self, inc, outc, scale=2):
        super(UpConv, self).__init__()
        self.scale = scale
        self.conv = nn.Conv2d(inc, outc, 3, stride=1, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True))


class ReBlock(nn.Module):
    def __init__(self, dim, rates):
        super(ReBlock, self).__init__()
        self.rates = rates
        for i, rate in enumerate(rates):
            self.__setattr__(
                "block{}".format(str(i).zfill(2)),
                nn.Sequential(
                    nn.ReflectionPad2d(rate), nn.Conv2d(dim, dim // 4, 3, padding=0, dilation=rate), nn.ReLU(True)
                ),
            )
        self.fuse = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))
        self.gate = nn.Sequential(nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, 3, padding=0, dilation=1))

    def forward(self, x):
        out = [self.__getattr__(f"block{str(i).zfill(2)}")(x) for i in range(len(self.rates))]
        out = torch.cat(out, 1)
        out = self.fuse(out)
        mask = my_layer_norm(self.gate(x))
        mask = torch.sigmoid(mask)
        return x * (1 - mask) + out * mask


def my_layer_norm(feat):
    mean = feat.mean((2, 3), keepdim=True)
    std = feat.std((2, 3), keepdim=True) + 1e-9
    feat = 2 * (feat - mean) / std - 1
    feat = 5 * feat
    return feat


# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(
        self,
    ):
        super(Discriminator, self).__init__()
        inc = 3
        self.conv = nn.Sequential(
            spectral_norm_fn(nn.Conv2d(inc, 64, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_fn(nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_fn(nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm_fn(nn.Conv2d(256, 512, 4, stride=1, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, stride=1, padding=1),
        )

        self.init_weights()

    def forward(self, x):
        feat = self.conv(x)
        return feat


class CoarseGenerator(nn.Module):
    def __init__(self, input_dim=3, cnum=16, batch_size=1, max_order=4, boundary='Dirichlet', use_cuda=True, device_ids=None):
        super(CoarseGenerator, self).__init__()
        # self.use_cuda = use_cuda
        # self.device_ids = device_ids

        self.conv1 = gen_GatedConv(input_dim, cnum * 1, 5, 1, padding=2, rate=1, norm='in', activation='elu')  # 256
        self.conv2 = gen_GatedConv(cnum * 1, cnum * 2, 3, 2, padding=1, rate=1, norm='in', activation='elu')  # 128
        self.conv3 = gen_GatedConv(cnum * 2, cnum * 4, 3, 2, padding=1, rate=1, norm='in', activation='elu')  # 64
        self.conv4 = gen_GatedConv(cnum * 4, cnum * 8, 3, 2, padding=1, rate=1, norm='in', activation='elu')  # 32

        #############
        # 中间进行PDE
        # PDE
        self.dt = 1e-2
        self.id = FD3d(kernel_size=[cnum * 8, 5, 5], max_order=0, dx=1 / (32 * 2.5), constraint='moment',
                       boundary=boundary)
        self.fd3d = FD3d(kernel_size=[cnum * 8, 5, 5], max_order=max_order, dx=1 / (32 * 2.5), constraint='moment',
                         boundary=boundary)
        N = self.fd3d.MomentBank.moment.size()[0]
        self._N = N
        x = 1 * np.arange(32) / 32
        sample = {}
        sample['x'] = np.repeat(x[np.newaxis, :], 32, axis=0)
        sample['y'] = np.repeat(x[:, np.newaxis], 32, axis=1)
        sample_torch = {}
        for k in sample:
            sample_torch[k] = torch.autograd.Variable(torch.tensor(sample[k]))

        xy = torch.stack([sample_torch['x'], sample_torch['y']], dim=2).unsqueeze(0).repeat(batch_size, 1, 1, 1)  # 原PDE中 x or y:[b, mesh_size[0], mesh_size[1]]  xy:[b, mesh_size[0], mesh_size[1], 2]
        # xy的扩展到batchsize
        for i in range(N):
            fitter = nn.Parameter(torch.zeros(1))
            self.register_buffer('coe' + str(i), fitter)

        self.timestep_embeded = TimeStepEncoding(num_hiddens=cnum * 8)
        self.symnet = SymNet(N)
        self.feature_embedded = nn.AdaptiveAvgPool2d((1, 1))
        self.FeaMapping = nn.Sequential(
            gen_linear(input_dim=cnum * 8, output_dim=cnum * 4, activation='elu'),
            gen_linear(input_dim=cnum * 4, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 4, activation='elu'),
            gen_linear(input_dim=cnum * 4, output_dim=cnum * 8, activation='elu'),
        )
        self.fea_style64 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 4 * 2, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )
        self.fea_style128 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 2 * 2, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )
        self.fea_style256 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 1 * 2, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )
        self.TimeMapping = nn.Sequential(
            gen_linear(input_dim=cnum * 8, output_dim=cnum * 4, activation='elu'),
            gen_linear(input_dim=cnum * 4, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 2, activation='elu'),
            gen_linear(input_dim=cnum * 2, output_dim=cnum * 4, activation='elu'),
            gen_linear(input_dim=cnum * 4, output_dim=cnum * 8, activation='elu'),
        )
        self.time_modulation64 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 8, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )
        self.time_modulation128 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )
        self.time_modulation256 = nn.Sequential(
            gen_conv(cnum * 8, cnum * 4, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='elu'),
            gen_conv(cnum * 4, cnum * 2, kernel_size=1, stride=1, padding=0, rate=1, norm='none',
                     activation='none'),
        )

        self.deconv1 = gen_conv(cnum * 8 * 2, cnum * 8, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv1_sty_conv1 = gen_sty_conv(cnum * 8, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_sty_conv2 = gen_sty_conv(cnum * 8 * 2, cnum * 8, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv1_gated = gen_GatedConv(cnum * 8 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2 = gen_conv(cnum * 4 * 2, cnum * 4, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv2_sty_conv1 = gen_sty_conv(cnum * 4, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_sty_conv2 = gen_sty_conv(cnum * 4 * 2, cnum * 4, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv2_gated = gen_GatedConv(cnum * 4 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3 = gen_conv(cnum * 2 * 2, cnum * 2, 4, 2, padding=1, rate=1, norm='in', activation='elu', transpose=True)
        self.deconv3_sty_conv1 = gen_sty_conv(cnum * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_sty_conv2 = gen_sty_conv(cnum * 2 * 2, cnum * 2, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.deconv3_gated = gen_GatedConv(cnum * 2 * 2, cnum * 1, 3, 1, padding=1, rate=1, norm='in', activation='elu')
        self.conv6 = gen_GatedConv(cnum * 1 * 2, cnum // 2, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv7 = gen_GatedConv(cnum // 2, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

        self.conv8 = gen_GatedConv(input_dim, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='elu')
        self.conv9 = gen_GatedConv(input_dim, input_dim, 3, 1, padding=1, rate=1, norm='none', activation='none')

    @property
    def coes(self):
        for i in range(self._N):
            yield self.__getattr__('coe' + str(i))

    def forward(self, x, timestep=2, frozen=False):
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        fea_embedding = self.feature_embedded(x_conv4)
        # ##############
        # PDE
        idkernel = self.id.MomentBank.kernel()
        fdkernel = self.fd3d.MomentBank.kernel()
        coe = []  # 把 N 个 filter 加进去
        for fitter in self.coes:
            # coe.append(fitter())
            coe.append(fitter.view(1, 1, 1, 1))
        # coe = torch.stack(coe, dim=1).unsqueeze(2)  # [b, N, h, w]
        coe = torch.stack(coe, dim=1)
        u = x_conv4
        for i in range(timestep):
            uid = self.id(u, idkernel).squeeze(1)
            ufd = self.fd3d(u, fdkernel)
            diff = coe * ufd
            u = uid + self.dt * self.symnet(diff)
        x_conv4 = u.view(x_conv4.size())
        ##############
        # 特征图 embedding
        batch_size = fea_embedding.size(0)
        fea_style_W = self.FeaMapping(fea_embedding.view(batch_size, -1))
        channel = fea_style_W.size(1)
        fea_style_W = fea_style_W.view(batch_size, channel, 1, 1)
        fea_style_64 = self.fea_style64(fea_style_W)
        fea_style_128 = self.fea_style128(fea_style_W)
        fea_style_256 = self.fea_style256(fea_style_W)

        style64 = fea_style_64
        style128 = fea_style_128
        style256 = fea_style_256
        style_64_scale, style_64_bias = torch.split(style64, style64.size(1) // 2, 1)
        style_128_scale, style_128_bias = torch.split(style128, style128.size(1) // 2, 1)
        style_256_scale, style_256_bias = torch.split(style256, style256.size(1) // 2, 1)

        time_embedding = self.timestep_embeded(timestep, device=x_conv4.device, frozen=frozen).flatten(
            start_dim=1)  # [1, cnum * 8]
        time_style_W = self.TimeMapping(time_embedding)
        time_style_W = time_style_W.view(1, time_style_W.size(1), 1, 1)
        time_mod64 = self.time_modulation64(time_style_W).permute(1, 0, 2, 3)
        time_mod128 = self.time_modulation128(time_style_W).permute(1, 0, 2, 3)
        time_mod256 = self.time_modulation256(time_style_W).permute(1, 0, 2, 3)

        tmp = self.deconv1(torch.cat([x_conv4, x_conv4], dim=1))
        x_deconv1_1 = self.deconv1_sty_conv1(tmp, time_mod64)
        tmp1 = F.interpolate(x_conv4, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv4, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv1_2 = self.deconv1_sty_conv2(torch.cat([tmp1, tmp2], dim=1), time_mod64)
        x_deconv1 = self.deconv1_gated(torch.cat([x_deconv1_1, x_deconv1_2], dim=1))
        x_deconv1 = style_64_scale * x_deconv1 + style_64_bias

        tmp = self.deconv2(torch.cat([x_deconv1, x_conv3], dim=1))
        x_deconv2_1 = self.deconv2_sty_conv1(tmp, time_mod128)
        tmp1 = F.interpolate(x_deconv1, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv3, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv2_2 = self.deconv2_sty_conv2(torch.cat([tmp1, tmp2], dim=1), time_mod128)
        x_deconv2 = self.deconv2_gated(torch.cat([x_deconv2_1, x_deconv2_2], dim=1))
        x_deconv2 = style_128_scale * x_deconv2 + style_128_bias

        tmp = self.deconv3(torch.cat([x_deconv2, x_conv2], dim=1))
        x_deconv3_1 = self.deconv3_sty_conv1(tmp, time_mod256)
        tmp1 = F.interpolate(x_deconv2, scale_factor=2, mode='bilinear', align_corners=True)
        tmp2 = F.interpolate(x_conv2, scale_factor=2, mode='bilinear', align_corners=True)
        x_deconv3_2 = self.deconv3_sty_conv2(torch.cat([tmp1, tmp2], dim=1), time_mod256)
        x_deconv3 = self.deconv3_gated(torch.cat([x_deconv3_1, x_deconv3_2], dim=1))
        x_deconv3 = style_256_scale * x_deconv3 + style_256_bias

        x_conv6 = self.conv6(torch.cat([x_deconv3, x_conv1], dim=1))
        x_conv7 = self.conv7(x_conv6)  # torch.Size([16, 3, 256, 256])
        x_conv8 = self.conv8(x_conv7)
        x_stage1 = self.conv9(x_conv8)  # torch.Size([16, 3, 256, 256])
        x_stage1 = nn.Tanh()(x_stage1)

        return x_stage1


class SymNet(nn.Module):
    def __init__(self, input_channels):
        super(SymNet, self).__init__()

        self.conv1 = nn.Conv3d(input_channels, 2, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(input_channels + 1, 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(input_channels + 2, 2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(input_channels + 3, 2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(input_channels + 4, 2, kernel_size=3, padding=1)
        self.output_layer = nn.Conv3d(input_channels + 5, 1, kernel_size=3, padding=1)

    def forward(self, diff):
        out1 = self.conv1(diff)
        a1 = out1[:, 0:1, :, :, :]
        b1 = out1[:, 1:2, :, :, :]
        f1 = torch.mul(a1, b1)
        out1 = torch.cat((diff, f1), dim=1)

        out2 = self.conv2(out1)
        a2 = out2[:, 0:1, :, :, :]
        b2 = out2[:, 1:2, :, :, :]
        f2 = torch.mul(a2, b2)
        out2 = torch.cat((out1, f2), dim=1)

        out3 = self.conv3(out2)
        a3 = out3[:, 0:1, :, :, :]
        b3 = out3[:, 1:2, :, :, :]
        f3 = torch.mul(a3, b3)
        out3 = torch.cat((out2, f3), dim=1)

        out4 = self.conv4(out3)
        a4 = out4[:, 0:1, :, :, :]
        b4 = out4[:, 1:2, :, :, :]
        f4 = torch.mul(a4, b4)
        out4 = torch.cat((out3, f4), dim=1)

        out5 = self.conv5(out4)
        a5 = out5[:, 0:1, :, :, :]
        b5 = out5[:, 1:2, :, :, :]
        f5 = torch.mul(a5, b5)
        out5 = torch.cat((out4, f5), dim=1)

        out = self.output_layer(out5)
        out = out.squeeze(dim=1)

        return out

def gen_linear(input_dim, output_dim, activation='lrelu'):
    return LinearBlock(input_dim=input_dim, output_dim=output_dim, activation=activation)

def gen_sty_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, norm='none',
                 activation='elu', transpose=False):
    return StyleConv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate, weight_norm='wn', norm=norm,
                       activation=activation, transpose=transpose)

def gen_conv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, norm='none',
             activation='elu', transpose=False):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate, norm=norm,
                       activation=activation, transpose=transpose)

def gen_GatedConv(input_dim, output_dim, kernel_size=3, stride=1, padding=0, rate=1, activation='elu', norm='none'):
    return GatedConv2d(input_dim, output_dim,
                 kernel_size, stride=stride,
                 conv_padding=padding, dilation=rate,
                 pad_type='zero',
                 activation=activation, norm=norm, sn=False)

class TimeStepEncoding(nn.Module):

    def __init__(self, num_hiddens, max_len=50):
        super().__init__()
        self.P = torch.zeros((1, max_len, num_hiddens))

        X = torch.arange(max_len, dtype=torch.float32).reshape(
            -1, 1) / torch.pow(10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, timestep, device, frozen=False):
        if frozen:
            embedding = self.P[:, 0, :].to(device)
        else:
            embedding = self.P[:, timestep-1, :].to(device)
        return embedding


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, activation='lrelu'):
        super(LinearBlock, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation.lower() == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation.lower() == 'prelu':
            self.activation = nn.PReLU()
        elif activation.lower() == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation.lower() == 'tanh':
            self.activation = nn.Tanh()
        elif activation.lower() == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation.lower() == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x


class StyleConv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(StyleConv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)


    def forward(self, x, scale):
        scale = scale.to(self.conv.weight.device)
        self.conv.weight = scale * self.conv.weight
        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1,
                 padding=0, conv_padding=0,
                 dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        self.use_bias = True
        # Initialize the padding scheme
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)  # inplace=True
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)  # inplace=True
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)  # inplace=True
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)  # inplace=True
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
            self.mask_conv2d = SpectralNorm(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=conv_padding, dilation=dilation, bias=self.use_bias)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)

        if self.activation:
            x = self.activation(conv) * gated_mask
        else:
            x = conv * gated_mask

        if self.norm:
            x = self.norm(x)

        return x

