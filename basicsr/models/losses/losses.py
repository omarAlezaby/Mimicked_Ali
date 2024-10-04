# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
from basicsr.models.archs.vgg_arch import VGGFeatureExtractor
import math


from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


# @weighted_loss
# def charbonnier_loss(pred, target, eps=1e-12):
#     return torch.sqrt((pred - target)**2 + eps)


class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', **kwards):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class PSNRLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', toY=False):
        super(PSNRLoss, self).__init__()
        assert reduction == 'mean'
        self.loss_weight = loss_weight
        self.scale = 10 / np.log(10)
        self.toY = toY
        self.coef = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1)
        self.first = True

    def forward(self, pred, target):
        assert len(pred.size()) == 4
        if self.toY:
            if self.first:
                self.coef = self.coef.to(pred.device)
                self.first = False

            pred = (pred * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.
            target = (target * self.coef).sum(dim=1).unsqueeze(dim=1) + 16.

            pred, target = pred / 255., target / 255.
            pass
        assert len(pred.size()) == 4

        return self.loss_weight * self.scale * torch.log(((pred - target) ** 2).mean(dim=(1, 2, 3)) + 1e-8).mean()

class FilterLoss(nn.Module): # kernel_size%2=1
	def __init__(self):
		super(FilterLoss, self).__init__()

	def forward(self, filter_weight):  # [out, in, kernel_size, kernel_size]
		weight = filter_weight
		out_c, in_c, k, k = weight.shape 
		index = torch.arange(-(k//2), k//2+1, 1)

		index = index.to(filter_weight.device)
		index = index.unsqueeze(dim=0).unsqueeze(dim=0)  # [1, 1, kernel_size] 
		index_i = index.unsqueeze(dim=3)  # [1, 1, kernel_size, 1]  
		index_j = index.unsqueeze(dim=0)  # [1, 1, 1, kernel_size]  

		diff = torch.mean(weight*index_i, dim=2).abs() + torch.mean(weight*index_j, dim=3).abs()
		return diff.mean()
     



def compute_cx(dist_tilde, band_width):
    w = torch.exp((1 - dist_tilde) / band_width)  # Eq(3)
    cx = w / torch.sum(w, dim=2, keepdim=True)  # Eq(4)
    return cx


def compute_relative_distance(dist_raw):
    dist_min, _ = torch.min(dist_raw, dim=2, keepdim=True)
    dist_tilde = dist_raw / (dist_min + 1e-5)
    return dist_tilde


def compute_cosine_distance(x, y):
    # mean shifting by channel-wise mean of `y`.
    y_mu = y.mean(dim=(0, 2, 3), keepdim=True)
    x_centered = x - y_mu
    y_centered = y - y_mu

    # L2 normalization
    x_normalized = F.normalize(x_centered, p=2, dim=1)
    y_normalized = F.normalize(y_centered, p=2, dim=1)

    # channel-wise vectorization
    N, C, *_ = x.size()
    x_normalized = x_normalized.reshape(N, C, -1)  # (N, C, H*W)
    y_normalized = y_normalized.reshape(N, C, -1)  # (N, C, H*W)

    # consine similarity
    cosine_sim = torch.bmm(x_normalized.transpose(1, 2),
                           y_normalized)  # (N, H*W, H*W)

    # convert to distance
    dist = 1 - cosine_sim
    return dist


def compute_l1_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)

    dist = x_vec.unsqueeze(2) - y_vec.unsqueeze(3)
    dist = dist.sum(dim=1).abs()
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_l2_distance(x, y):
    N, C, H, W = x.size()
    x_vec = x.view(N, C, -1)
    y_vec = y.view(N, C, -1)
    x_s = torch.sum(x_vec ** 2, dim=1, keepdim=True)
    y_s = torch.sum(y_vec ** 2, dim=1, keepdim=True)

    A = y_vec.transpose(1, 2) @ x_vec
    dist = y_s - 2 * A + x_s.transpose(1, 2)
    dist = dist.transpose(1, 2).reshape(N, H * W, H * W)
    dist = dist.clamp(min=0.)

    return dist


def compute_meshgrid(shape):
    N, C, H, W = shape
    rows = torch.arange(0, H, dtype=torch.float32) / (H + 1)
    cols = torch.arange(0, W, dtype=torch.float32) / (W + 1)

    feature_grid = torch.meshgrid(rows, cols)
    feature_grid = torch.stack(feature_grid).unsqueeze(0)
    feature_grid = torch.cat([feature_grid for _ in range(N)], dim=0)

    return feature_grid

@weighted_loss
def contextual_loss(pred, target, band_width=0.5, loss_type='cosine'):
    """
    Computes contepredtual loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)

    dist_tilde = compute_relative_distance(dist_raw)
    cx = compute_cx(dist_tilde, band_width)
    cx = torch.mean(torch.max(cx, dim=1)[0], dim=1)  # Eq(1)
    cx_loss = -torch.log(cx + 1e-5)  # Eq(5)
    # pdb.set_trace()
    return cx_loss


class ContextualLoss(nn.Module):
    """
    Creates a criterion that measures the contextual loss.

    Args
    ---
    band_width : int, optional
        a band_width parameter described as :math:`h` in the paper.
    use_vgg : bool, optional
        if you want to use VGG feature, set this `True`.
    vgg_layer : str, optional
        intermidiate layer name for VGG feature.
        Now we support layer names:
            `['relu1_2', 'relu2_2', 'relu3_4', 'relu4_4', 'relu5_4']`
    """

    def __init__(self, band_width=0.5, loss_type='cosine',
                 use_vgg=True, vgg_layers=['conv4_4'],
                 loss_weight=1.0, reduction='mean'):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {reduction}.')

        assert band_width > 0, 'band_width parameter must be positive.'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=vgg_layers,
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'

        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        cx_loss = 0
        for k in pred_features.keys():
            # cx_loss += contextual_loss(pred_features[k], target_features[k],
            #                            band_width=self.band_width, loss_type=self.loss_type,
            #                            weight=weight, reduction=self.reduction)
            cx_loss += contextual_loss(target_features[k], pred_features[k],
                                       band_width=self.band_width, loss_type=self.loss_type,
                                       weight=None, reduction=self.reduction)

        cx_loss *= self.loss_weight
        return cx_loss

@weighted_loss
def cobi_loss(pred, target, weight_sp=0.1, band_width=0.5, loss_type='cosine'):
    """
    Computes CoBi loss between pred and target.
    Parameters
    ---
    pred : torch.Tensor
        features of shape (N, C, H, W).
    target : torch.Tensor
        features of shape (N, C, H, W).
    weight_sp : float, optional
        a balancing weight between spatial and feature loss.
    band_width : float, optional
        a band-width parameter used to convert distance to similarity.
        in the paper, this is described as :math:`h`.
    loss_type : str, optional
        a loss type to measure the distance between features.
        Note: `l1` and `l2` frequently raises OOM.
    Returns
    ---
    cx_loss : torch.Tensor
        contextual loss between x and y (Eq (1) in the paper)
    """

    assert pred.size() == target.size(), 'input tensor must have the same size.'
    assert loss_type in ['cosine', 'l1', 'l2'], f"select a loss type from \
                                                {['cosine', 'l1', 'l2']}."

    N, C, H, W = pred.size()

    # spatial loss
    grid = compute_meshgrid(pred.shape).to(pred.device)
    dist_raw = compute_l2_distance(grid, grid)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_sp = compute_cx(dist_tilde, band_width)

    # feature loss
    if loss_type == 'cosine':
        dist_raw = compute_cosine_distance(pred, target)
    elif loss_type == 'l1':
        dist_raw = compute_l1_distance(pred, target)
    elif loss_type == 'l2':
        dist_raw = compute_l2_distance(pred, target)
    dist_tilde = compute_relative_distance(dist_raw)
    cx_feat = compute_cx(dist_tilde, band_width)

    # combine loss
    cx_combine = (1 - weight_sp) * cx_feat + weight_sp * cx_sp

    k_max_NC, _ = torch.max(cx_combine, dim=2, keepdim=True)

    cx = k_max_NC.mean(dim=1)
    cx_loss = -torch.log(cx + 1e-5)

    return cx_loss


class CoBiLoss(nn.Module):
    """
    Creates a criterion that measures the boci loss.
    """

    def __init__(self, band_width=0.5, weight_sp=0.1, loss_type='cosine',
                 use_vgg=True, vgg_layers=['conv4_4'],
                 loss_weight=1.0, reduction='mean', use_rgb=False, rgb_weight = .1, rgb_kernal=10):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. ' f'Supported ones are: {_reduction_modes}')
        if loss_type not in ['cosine', 'l1', 'l2']:
            raise ValueError(f'Unsupported loss mode: {reduction}.')

        assert band_width > 0, 'band_width parameter must be positive.'
        assert weight_sp >= 0 and weight_sp <= 1, 'weight_sp out of range [0, 1].'

        self.band_width = band_width
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss_type = loss_type
        self.weight_sp = weight_sp
        self.use_vgg = use_vgg
        self.rgb_kernal = rgb_kernal
        self.use_rgb = use_rgb
        self.rgb_weight = rgb_weight

        if use_vgg:
            self.vgg_model = VGGFeatureExtractor(
                layer_name_list=vgg_layers,
                vgg_type='vgg19')

    def forward(self, pred, target, weight=None, **kwargs):
        assert hasattr(self, 'vgg_model'), 'Please specify VGG model.'
        assert pred.shape[1] == 3 and target.shape[1] == 3,\
            'VGG model takes 3 chennel images.'
        
        # picking up vgg feature maps
        pred_features = self.vgg_model(pred)
        target_features = self.vgg_model(target.detach())

        if self.use_rgb:
            pred_features_rgb = torch.nn.functional.unfold(pred, self.rgb_kernal, stride=self.rgb_kernal)
            N, C, L = pred_features_rgb.size()
            pred_features_rgb = pred_features_rgb.view(N,C, int(math.sqrt(L)), int(math.sqrt(L)))
            target_features_rgb = torch.nn.functional.unfold(target.detach(), self.rgb_kernal, stride=self.rgb_kernal)
            target_features_rgb = target_features_rgb.view(N,C, int(math.sqrt(L)), int(math.sqrt(L)))

        cx_loss = 0
        for k in pred_features.keys():
            cx_loss += cobi_loss(pred_features[k], target_features[k],
                                 weight_sp=self.weight_sp, band_width=self.band_width,
                                 loss_type=self.loss_type, weight=weight, reduction=self.reduction)

        if self.use_rgb:
            cx_loss += self.rgb_weight * cobi_loss(pred_features_rgb, target_features_rgb,
                                 weight_sp=self.weight_sp, band_width=self.band_width,
                                 loss_type=self.loss_type, weight=weight, reduction=self.reduction)
            
        cx_loss *= self.loss_weight
        return cx_loss
