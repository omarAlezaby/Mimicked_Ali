
import torch
import torch.nn as nn
import torch.nn.functional as F
from builtins import print
import math
import pdb
from copy import deepcopy
from basicsr.models.archs.pwcnet_arch import PWCNET
import torch.nn.functional as F

class L1LiteISPLoss(torch.nn.Module):
    def __init__(self, super_res_ali = False, mim=False, luminance = False, loss_weight = 1.0, reduction = 'mean', **kwargs):
        super(L1LiteISPLoss, self).__init__()
        self.L1_loss = torch.nn.L1Loss()

        self.backwarp_tenGrid = {}
        self.backwarp_tenPartial = {}

        self.flow_net = PWCNET()
        self.set_requires_grad(self.flow_net, requires_grad=False)
        self.super_res = super_res_ali
        self.mim = mim
        self.luminance = luminance
        self.loss_weight = loss_weight

    def forward(self, inp, out, ref, weight=None, **kwargs):
        
        N, C, H, W = inp.size()
        if self.super_res:
            down_dslr = F.interpolate(input=ref, size=(H, W), 
									  mode='bilinear', align_corners=True)
            flow = self.get_flow(inp, down_dslr, self.flow_net)
            up_flow = F.interpolate(input=flow, size=(H*4, W*4), 
                                mode='bilinear', align_corners=True) * 4.
            gt_warp, align_mask = self.get_backwarp('/', ref, self.flow_net, up_flow)
            out_masked = out * align_mask

        elif self.mim:
            N, C, H, W = ref.size()
            down_dslr = F.interpolate(input=inp, size=(H, W), 
									  mode='bilinear', align_corners=True)
            flow = self.get_flow(down_dslr, ref, self.flow_net)

            gt_warp, align_mask = self.get_backwarp('/', ref, self.flow_net, flow)
            out_masked = out * align_mask

        else: 
            gt_warp, align_mask = self.get_backwarp(out, ref, self.flow_net)
            out_masked = out * align_mask

        
        if self.luminance:
            out_masked_Y = 0.299 * out_masked[:,0]  + 0.587 * out_masked[:,1] + 0.114 * out_masked[:,2]
            gt_warp_Y = 0.299 * gt_warp[:,0]  + 0.587 * gt_warp[:,1] + 0.114 * gt_warp[:,2]

            down_dslr_Pb = -0.168736 * down_dslr[:,0] - 0.331264* down_dslr[:,1] + 0.5* down_dslr[:,2]
            down_dslr_Pr = 0.5*down_dslr[:,0] - 0.418688*down_dslr[:,1] - 0.081312*down_dslr[:,2]

            out_Pb = -0.168736 * out[:,0] - 0.331264* out[:,1] + 0.5* out[:,2]
            out_Pr = 0.5*out[:,0] - 0.418688*out[:,1] - 0.081312*out[:,2]

            loss_algin = self.L1_loss(out_masked_Y, gt_warp_Y)
            loss_algin += self.L1_loss(down_dslr_Pb, out_Pb)
            loss_algin += self.L1_loss(down_dslr_Pr, out_Pr)
            
        else :
            loss_algin = self.L1_loss(out_masked, gt_warp)

        return self.loss_weight * loss_algin
    
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
                         
    def estimate(self, tenFirst, tenSecond, net):
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
                         input=net(tenPreprocessedFirst, tenPreprocessedSecond), 
                         size=(intHeight, intWidth), mode='bilinear', align_corners=False)

        tenFlow[:, 0, :, :] *= float(intWidth) / float(intPreprocessedWidth)
        tenFlow[:, 1, :, :] *= float(intHeight) / float(intPreprocessedHeight)

        return tenFlow[:, :, :, :]
    
    def backwarp(self, tenInput, tenFlow):
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

        return tenOutput

    def get_backwarp(self, tenFirst, tenSecond, net, flow=None):
        if flow is None:
            flow = self.get_flow(tenFirst, tenSecond, net)
        
        tenoutput = self.backwarp(tenSecond, flow)     
        tenMask = tenoutput[:, -1:, :, :]
        tenMask[tenMask > 0.999] = 1.0
        tenMask[tenMask < 1.0] = 0.0
        return tenoutput[:, :-1, :, :] * tenMask, tenMask
    
    def get_flow(self, tenFirst, tenSecond, net):
        with torch.no_grad():
            net.eval()
            flow = self.estimate(tenFirst, tenSecond, net) 
        return flow