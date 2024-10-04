import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class DownBlock(nn.Module):
	def __init__(self, scale):
		super().__init__()
		self.scale = scale

	def forward(self, x):
		n, c, h, w = x.size()
		x = x.view(n, c, h//self.scale, self.scale, w//self.scale, self.scale)
		x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
		x = x.view(n, c*(self.scale**2), h//self.scale, w//self.scale)
		return x
	
class MimSimp(nn.Module):  # Auxiliary-LR Generator
	def __init__(self, **kwards):
		super(MimSimp, self).__init__()
		n_feats = 64


		self.head = conv(3, n_feats, kernel_size=1, stride=1, padding=0, mode='CR')

		self.conv_7x7 = conv(n_feats, n_feats, kernel_size=7, stride=1, padding=3, mode='CR')
		self.conv_5x5 = conv(n_feats, n_feats, kernel_size=5, stride=1, padding=2, mode='CR')
		self.conv_3x3 = conv(n_feats, n_feats, kernel_size=3, stride=1, padding=1, mode='CR')

		self.conv_1x1 = conv(n_feats, n_feats, kernel_size=1, stride=1, padding=0, mode='CRCRCR')

		self.tail = conv(n_feats, 3, kernel_size=1, stride=1, padding=0, mode='C')

		self.guide_net = seq(
			conv(3+3, n_feats, 7, stride=2, padding=0, mode='CR'),
			conv(n_feats, n_feats, kernel_size=3, stride=1, padding=1, mode='CRCRC'),
			nn.AdaptiveAvgPool2d(1),
			conv(n_feats, n_feats, 1, stride=1, padding=0, mode='C')
		)

	def forward(self, hr, lr):
		guide = self.guide_net(torch.cat([hr, lr], dim=1))

		head = self.head(hr)
		out = head * guide + head

		out = self.conv_3x3(self.conv_5x5(self.conv_7x7(out)))
		out = self.conv_1x1(out) + head
		out = self.tail(out)

		return out


def seq(*args):
	if len(args) == 1:
		args = args[0]
	if isinstance(args, nn.Module):
		return args
	modules = OrderedDict()
	if isinstance(args, OrderedDict):
		for k, v in args.items():
			modules[k] = seq(v)
		return nn.Sequential(modules)
	assert isinstance(args, (list, tuple))
	return nn.Sequential(*[seq(i) for i in args])

def conv(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
		 output_padding=0, dilation=1, groups=1, bias=True,
		 padding_mode='zeros', mode='CBR'):
	L = []
	for t in mode:
		if t == 'C':
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=groups,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'X':
			assert in_channels == out_channels
			L.append(nn.Conv2d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=kernel_size,
							   stride=stride,
							   padding=padding,
							   dilation=dilation,
							   groups=in_channels,
							   bias=bias,
							   padding_mode=padding_mode))
		elif t == 'T':
			L.append(nn.ConvTranspose2d(in_channels=in_channels,
										out_channels=out_channels,
										kernel_size=kernel_size,
										stride=stride,
										padding=padding,
										output_padding=output_padding,
										groups=groups,
										bias=bias,
										dilation=dilation,
										padding_mode=padding_mode))
		elif t == 'B':
			L.append(nn.BatchNorm2d(out_channels))
		elif t == 'I':
			L.append(nn.InstanceNorm2d(out_channels, affine=True))
		elif t == 'i':
			L.append(nn.InstanceNorm2d(out_channels))
		elif t == 'R':
			L.append(nn.ReLU(inplace=True))
		elif t == 'r':
			L.append(nn.ReLU(inplace=False))
		elif t == 'S':
			L.append(nn.Sigmoid())
		elif t == 'P':
			L.append(nn.PReLU())
		elif t == 'L':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
		elif t == 'l':
			L.append(nn.LeakyReLU(negative_slope=0.2, inplace=False))
		elif t == '2':
			L.append(nn.PixelShuffle(upscale_factor=2))
		elif t == '3':
			L.append(nn.PixelShuffle(upscale_factor=3))
		elif t == '4':
			L.append(nn.PixelShuffle(upscale_factor=4))
		elif t == 'U':
			L.append(nn.Upsample(scale_factor=2, mode='nearest'))
		elif t == 'u':
			L.append(nn.Upsample(scale_factor=3, mode='nearest'))
		elif t == 'M':
			L.append(nn.MaxPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		elif t == 'A':
			L.append(nn.AvgPool2d(kernel_size=kernel_size,
								  stride=stride,
								  padding=0))
		else:
			raise NotImplementedError('Undefined type: '.format(t))
	return seq(*L)


class MeanShift(nn.Conv2d):
	""" is implemented via group conv """
	def __init__(self, rgb_range=1, rgb_mean=(0.4488, 0.4371, 0.4040),
				 rgb_std=(1.0, 1.0, 1.0), sign=-1):
		super(MeanShift, self).__init__(3, 3, kernel_size=1, groups=3)
		std = torch.Tensor(rgb_std)
		self.weight.data = torch.ones(3).view(3, 1, 1, 1) / std.view(3, 1, 1, 1)
		self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
		for p in self.parameters():
			p.requires_grad = False

if __name__ == '__main__':
    img_channel = 3
    width = 32

    # enc_blks = [2, 2, 4, 8]
    # middle_blk_num = 12
    # dec_blks = [2, 2, 2, 2]
    net = AuxSimp()


    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)