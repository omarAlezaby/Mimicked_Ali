"""
implementation of the PWC-DC network for optical flow estimation by Sun et al., 2018

Jinwei Gu and Zhile Ren
"""

import math
import pdb
import numpy as np
from numpy.core.fromnumeric import size
import os
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

from basicsr.models.archs.correlation import correlation # the custom cost volume layer
import pickle


# Borrow the code of the optical flow network (PWC-Net) from https://github.com/sniklaus/pytorch-pwc/
class PWCNET(torch.nn.Module):
	def __init__(self):
		super(PWCNET, self).__init__()

		class Extractor(torch.nn.Module):
			def __init__(self):
				super(Extractor, self).__init__()

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

			def forward(self, tenInput):
				tenOne = self.netOne(tenInput)
				tenTwo = self.netTwo(tenOne)
				tenThr = self.netThr(tenTwo)
				tenFou = self.netFou(tenThr)
				tenFiv = self.netFiv(tenFou)
				tenSix = self.netSix(tenFiv)

				return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]

		class Decoder(torch.nn.Module):
			def __init__(self, intLevel):
				super(Decoder, self).__init__()

				intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 
								81 + 128 + 2 + 2, 81, None ][intLevel + 1]
				intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 
							   81 + 128 + 2 + 2, 81, None ][intLevel + 0]
				
				self.backwarp_tenGrid = {}
				self.backwarp_tenPartial = {}

				if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, 
												  out_channels=2, kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(
												  in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, 
												  kernel_size=4, stride=2, padding=1)
				if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

				self.netOne = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netTwo = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netThr = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96,
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFou = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netFiv = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, 
									kernel_size=3, stride=1, padding=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
				)

				self.netSix = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, 
									kernel_size=3, stride=1, padding=1)
				)

			def forward(self, tenFirst, tenSecond, objPrevious):
				tenFlow = None
				tenFeat = None

				if objPrevious is None:
					tenFlow = None
					tenFeat = None
					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(
								tenFirst=tenFirst, tenSecond=tenSecond), negative_slope=0.1, inplace=False)
					tenFeat = torch.cat([ tenVolume ], 1)

				elif objPrevious is not None:
					tenFlow = self.netUpflow(objPrevious['tenFlow'])
					tenFeat = self.netUpfeat(objPrevious['tenFeat'])

					tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(
								tenFirst=tenFirst, tenSecond=self.backwarp(tenInput=tenSecond, 
								tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

					tenFeat = torch.cat([ tenVolume, tenFirst, tenFlow, tenFeat ], 1)

				tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
				tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

				tenFlow = self.netSix(tenFeat)

				return {
					'tenFlow': tenFlow,
					'tenFeat': tenFeat
				}

			def backwarp(self, tenInput, tenFlow):
				index = str(tenFlow.shape) + str(tenInput.device)
				if index not in self.backwarp_tenGrid:
					tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
											tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
					tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
											tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
					self.backwarp_tenGrid[index] = torch.cat([ tenHor, tenVer ], 1).to(tenInput.device)

				if index not in self.backwarp_tenPartial:
					self.backwarp_tenPartial[index] = tenFlow.new_ones([ tenFlow.shape[0], 
															1, tenFlow.shape[2], tenFlow.shape[3] ])

				tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
									tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
				tenInput = torch.cat([ tenInput, self.backwarp_tenPartial[index] ], 1)

				tenOutput = torch.nn.functional.grid_sample(input=tenInput, 
							grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
							mode='bilinear', padding_mode='zeros', align_corners=False)

				tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0

				return tenOutput[:, :-1, :, :] * tenMask

		class Refiner(torch.nn.Module):
			def __init__(self):
				super(Refiner, self).__init__()
				self.netMain = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, 
									out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
					torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
					torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
				)

			def forward(self, tenInput):
				return self.netMain(tenInput)

		self.netExtractor = Extractor()

		self.netTwo = Decoder(2)
		self.netThr = Decoder(3)
		self.netFou = Decoder(4)
		self.netFiv = Decoder(5)
		self.netSix = Decoder(6)

		self.netRefiner = Refiner()

		# pickle.load = partial(pickle.load, encoding="latin1")
		# pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")

		self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight 
							   in torch.load('experiments/pretrained_models/pwc-net.pth', map_location=lambda storage, 
							   loc: storage, pickle_module=pickle).items() })

	def forward(self, tenFirst, tenSecond):
		tenFirst = self.netExtractor(tenFirst)
		tenSecond = self.netExtractor(tenSecond)
		objEstimate = self.netSix(tenFirst[-1], tenSecond[-1], None)
		objEstimate = self.netFiv(tenFirst[-2], tenSecond[-2], objEstimate)
		objEstimate = self.netFou(tenFirst[-3], tenSecond[-3], objEstimate)
		objEstimate = self.netThr(tenFirst[-4], tenSecond[-4], objEstimate)
		objEstimate = self.netTwo(tenFirst[-5], tenSecond[-5], objEstimate)
		return objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])



class FlowGenerator(nn.Module):
    """PWC-DC net for flow generation.

    Args:
        path (str): Pre-trained path. Default: None.
        requires_grad (bool): If true, the parameters of VGG network will be
            optimized. Default: False.
    """

    def __init__(self,
                 path=None,
                 requires_grad=False,):
        super().__init__()

        self.model = PWCNET()

        if not requires_grad:
            self.model.eval()
            for param in self.parameters():
                param.requires_grad = False
        else:
            self.model.train()
            for param in self.parameters():
                param.requires_grad = True

        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}


    def forward(self):

        return

    def estimate(self, tenFirst, tenSecond):
        assert(tenFirst.shape[3] == tenSecond.shape[3])
        assert(tenFirst.shape[2] == tenSecond.shape[2])
        intWidth = tenFirst.shape[3]
        intHeight = tenFirst.shape[2]
        # tenPreprocessedFirst = tenFirst.view(1, 3, intHeight, intWidth)
        # tenPreprocessedSecond = tenSecond.view(1, 3, intHeight, intWidth)

        intPreprocessedWidth = int(math.floor(math.ceil(intWidth / 64.0) * 64.0))
        intPreprocessedHeight = int(math.floor(math.ceil(intHeight / 64.0) * 64.0))

        tenPreprocessedFirst = F.interpolate(input=tenFirst, 
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)
        tenPreprocessedSecond = F.interpolate(input=tenSecond,
                                size=(intPreprocessedHeight, intPreprocessedWidth), 
                                mode='bilinear', align_corners=False)

        tenFlow = 20.0 * F.interpolate(
                         input=self.model(tenPreprocessedFirst, tenPreprocessedSecond), 
                         size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[:, :, :, :]


    def algin_features(self, tenInput, tenFlow):
        index = str(tenFlow.shape) + str(tenInput.device)
        if index not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), 
                     tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), 
                     tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])
            self.backwarp_tenGrid[index] = torch.cat([tenHor, tenVer], 1).to(tenInput.device)

        if index not in self.backwarp_tenPartial:
            self.backwarp_tenPartial[index] = tenFlow.new_ones([
                 tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3]])

        tenFlow = torch.cat([tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), 
                             tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0)], 1)
        tenInput = torch.cat([tenInput, self.backwarp_tenPartial[index]], 1)

        tenOutput = F.grid_sample(input=tenInput, 
                    grid=(self.backwarp_tenGrid[index] + tenFlow).permute(0, 2, 3, 1), 
                    mode='bilinear', padding_mode='zeros', align_corners=False)
        
        tenMask = tenOutput[:, -1:, :, :]
        tenMask[tenMask > 0.999] = 1.0
        tenMask[tenMask < 1.0] = 0.0

        return tenMask, tenOutput[:, :-1, :, :]
    
    def get_flow(self, tenFirst, tenSecond):
        with torch.no_grad():
            flow = self.estimate(tenFirst, tenSecond) 
        return flow
    
    def resize_flow(self, flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
        """Resize a flow according to ratio or shape.

        Args:
            flow (Tensor): Precomputed flow. shape [N, 2, H, W].
            size_type (str): 'ratio' or 'shape'.
            sizes (list[int | float]): the ratio for resizing or the final output
                shape.
                1) The order of ratio should be [ratio_h, ratio_w]. For
                downsampling, the ratio should be smaller than 1.0 (i.e., ratio
                < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
                ratio > 1.0).
                2) The order of output_size should be [out_h, out_w].
            interp_mode (str): The mode of interpolation for resizing.
                Default: 'bilinear'.
            align_corners (bool): Whether align corners. Default: False.

        Returns:
            Tensor: Resized flow.
        """
        _, _, flow_h, flow_w = flow.size()
        if size_type == 'ratio':
            output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
        elif size_type == 'shape':
            output_h, output_w = sizes[0], sizes[1]
        else:
            raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

        input_flow = flow.clone()
        ratio_h = output_h / flow_h
        ratio_w = output_w / flow_w
        input_flow[:, 0, :, :] *= ratio_w
        input_flow[:, 1, :, :] *= ratio_h
        resized_flow = F.interpolate(
            input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
        return resized_flow


if __name__ == '__main__':
    h, w = 256, 256
    model = PWCNET().cuda()
    model.eval()
    print(model)

    x = torch.randn((1, 3, h, w)).cuda()
    y = torch.randn((1, 3, h, w)).cuda()
    with torch.no_grad():
        out = model(x, y)
    print(out.shape)


    
