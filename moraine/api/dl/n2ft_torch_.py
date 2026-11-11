"""Noise2Fringe Transformer (n2ft) model in torch"""


__all__ = ['N2FT', 'ChannelAffine', 'fold_batch_norms']

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAffine(nn.Module):
    """(x - mean) * scale + bias per channel (dimension 1, like BatchNorm1d): what a BatchNorm1d computes in evaluation
    mode, in the same order of operations (folding the mean into the bias loses precision for large means)"""
    def __init__(self, mean, scale, bias):
        super().__init__()
        self.register_buffer('mean', mean)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', bias)

    def forward(self, x):
        shape = (1, -1) + (1,)*(x.dim()-2)
        return torch.addcmul(self.bias.view(shape), x - self.mean.view(shape), self.scale.view(shape))

def fold_batch_norms(model):
    """replace the BatchNorm1d layers of a model in evaluation mode by the per channel affine function they compute"""
    for name, module in list(model.named_children()):
        if isinstance(module, nn.BatchNorm1d):
            scale = module.weight/torch.sqrt(module.running_var + module.eps)
            setattr(model, name, ChannelAffine(module.running_mean.detach().clone(), scale.detach().clone(),
                                               module.bias.detach().clone()))
        else:
            fold_batch_norms(module)
    return model

def group_features(features, idx):
    B, N, C = features.shape
    _, Nf, k = idx.shape

    gathered = features[
        torch.arange(B, device=features.device).view(B, 1, 1),
        idx,
        :,
    ]
    return gathered

class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        self.linear_q = nn.Linear(in_planes, out_planes)
        self.linear_k = nn.Linear(in_planes, out_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)

        self.linear_p = nn.Sequential(
            nn.Linear(2, 2),
            nn.BatchNorm1d(2),
            nn.ReLU(inplace=True),
            nn.Linear(2, out_planes)
        )

        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(out_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes, out_planes // share_planes),
            nn.BatchNorm1d(out_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // share_planes, out_planes // share_planes)
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, pos, x, idx) -> torch.Tensor:
        idx = idx[:,:,:self.nsample]
        batch_size, num_points, nsample = idx.shape

        x_q = self.linear_q(x)
        x_k = self.linear_k(x)
        x_v = self.linear_v(x)

        x_k = group_features(x_k,idx)
        x_v = group_features(x_v,idx)
        p_r = group_features(pos,idx)

        p_r = p_r - pos.unsqueeze(2)

        p_r = p_r.view(-1, nsample, 2)
        for i, layer in enumerate(self.linear_p):
            if isinstance(layer, (nn.BatchNorm1d, ChannelAffine)):
                p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()
            else:
                p_r = layer(p_r)
        p_r = p_r.view(batch_size, num_points, nsample, self.out_planes)

        w = x_k - x_q.unsqueeze(2) + p_r

        w = w.view(batch_size*num_points,nsample,self.out_planes)
        for i, layer in enumerate(self.linear_w):
            if isinstance(layer, (nn.BatchNorm1d, ChannelAffine)):
                w = layer(w.transpose(1, 2).contiguous()).transpose(1,2).contiguous()
            else:
                w = layer(w)
        w = w.view(batch_size,num_points,nsample,self.out_planes//self.share_planes)

        w = self.softmax(w) 

        s = self.share_planes
        x = ((x_v + p_r).view(
            batch_size, num_points, nsample, self.share_planes, self.out_planes//self.share_planes)
             * w.unsqueeze(3)).sum(2).view(batch_size, num_points, self.out_planes)

        return x

class PointTransformerBlock(nn.Module):
    def __init__(self, in_planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, in_planes, bias=False)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.transformer2 = PointTransformerLayer(in_planes, in_planes, share_planes, nsample)
        self.bn2 = nn.BatchNorm1d(in_planes)
        self.linear3 = nn.Linear(in_planes, in_planes, bias=False)
        self.bn3 = nn.BatchNorm1d(in_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x, idx):
        batch_size, num_points, in_planes = x.shape 
        identity = x
        x = self.linear1(x).view(batch_size*num_points,in_planes)
        x = self.relu(self.bn1(x)).view(batch_size, num_points, in_planes)

        x = self.transformer2(pos, x, idx).view(batch_size*num_points,in_planes)
        x = self.relu(self.bn2(x)).view(batch_size, num_points, in_planes)

        x = self.linear3(x).view(batch_size*num_points,in_planes)
        x = self.bn3(x).view(batch_size, num_points, in_planes)

        x += identity
        x = self.relu(x)
        return x

class BlockSequence(nn.Module):
    def __init__(self, block, planes, blocks, share_planes=8, nsample=16):
        super().__init__()
        self.layers = nn.ModuleList([
            block(planes, share_planes, nsample=nsample) for _ in range(blocks)
        ])

    def forward(self, pos, x, idx):
        for layer in self.layers:
            x = layer(pos, x, idx)
        return x

class MLP(nn.Module):
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.linear = nn.Linear(2+in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x):
        batch_size, num_points, in_planes = x.shape
        x = torch.cat((pos, x),-1)
        x = self.linear(x).view(batch_size*num_points,self.out_planes)
        x = self.relu(self.bn(x)).view(batch_size, num_points, self.out_planes)
        return x

class TransitionDown(nn.Module):
    def __init__(self, in_planes, out_planes, nsample=16):
        super().__init__()
        self.nsample = nsample
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.linear = nn.Linear(2+in_planes, out_planes, bias=False)
        self.pool = nn.MaxPool1d(nsample)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x, idx):
        idx = idx[:,:,:self.nsample]
        batch_size, num_points, nsample = idx.shape
        x = torch.cat((pos, x),-1)
        x = group_features(x, idx)
        x = x.view(batch_size*num_points,nsample,2+self.in_planes)

        x = self.linear(x)
        x = self.bn(x.transpose(1,2).contiguous())
        x = self.pool(self.relu(x))

        x = x.view(batch_size, num_points, self.out_planes)
        return x

class TransitionUp(nn.Module):
    def __init__(self, x1_planes, x2_planes, out_planes):
        super().__init__()
        self.linear = nn.Linear(x1_planes+x2_planes+2,out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pos, x1, x2, ii, ww):
        batch_size, num_points = ii.shape[:2]

        x1 = group_features(x1, ii)
        x1 = torch.sum(x1*ww.unsqueeze(-1),dim=2)
        x = torch.cat([x1, x2, pos], dim=-1)

        x = x.view(batch_size*num_points,-1)
        x = self.relu(self.bn(self.linear(x)))
        x = x.view(batch_size, num_points, -1)
        return x

def slice_points(pos: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    B = pos.size(0)
    return pos[torch.arange(B, device=pos.device).view(B, 1, 1),
               idx.unsqueeze(2),
               torch.arange(2, device=pos.device).view(1, 1, 2),
    ]

class N2FT(nn.Module):
    def __init__(self, block, blocks):
        super().__init__()
        planes = [32, 64, 128, 256, 512]
        share_planes = 8
        nsample = [8, 16, 16, 16, 16]

        self.down0 = MLP(2,planes[0])
        self.enc0 = self._make_block(block, planes[0], blocks[0], share_planes, nsample=nsample[0])
        self.down1 = TransitionDown(planes[0],planes[1],nsample[1])
        self.enc1 = self._make_block(block, planes[1], blocks[1], share_planes, nsample=nsample[1])
        self.down2 = TransitionDown(planes[1],planes[2],nsample[2])
        self.enc2 = self._make_block(block, planes[2], blocks[2], share_planes, nsample=nsample[2])
        self.down3 = TransitionDown(planes[2],planes[3],nsample[3])
        self.enc3 = self._make_block(block, planes[3], blocks[3], share_planes, nsample=nsample[3])

        self.up2 = TransitionUp(planes[3], planes[2], planes[2])
        self.dec2 = self._make_block(block, planes[2], 1, share_planes, nsample=nsample[2])
        self.up1 = TransitionUp(planes[2], planes[1], planes[1])
        self.dec1 = self._make_block(block, planes[1], 1, share_planes, nsample=nsample[1])
        self.up0 = TransitionUp(planes[1], planes[0], planes[0])
        self.dec0 = self._make_block(block, planes[0], 1, share_planes, nsample=nsample[0])

        self.out_layer = nn.Sequential(nn.Linear(planes[0], planes[0]), nn.ReLU(inplace=True), nn.Linear(planes[0], 2))

    def _make_block(self, block, planes, blocks, share_planes=8, nsample=16):
        return BlockSequence(block, planes, blocks, share_planes, nsample)

    def forward(
        self, pos, x,
        ii00, ii01, ii11, ii12, ii22, ii23, ii33,
        ii10, ii21, ii32,
        ww10, ww21, ww32
    ): 

        pos0 = pos
        pos1 = slice_points(pos0, ii01[:, :, 0])
        pos2 = slice_points(pos1, ii12[:, :, 0])
        pos3 = slice_points(pos2, ii23[:, :, 0])

        l0 = self.down0(pos, x)
        l0 = self.enc0(pos0, l0, ii00)

        l1 = self.down1(pos0, l0, ii01)
        l1 = self.enc1(pos1, l1, ii11)

        l2 = self.down2(pos1, l1, ii12)
        l2 = self.enc2(pos2, l2, ii22)

        l3 = self.down3(pos2, l2, ii23)
        l3 = self.enc3(pos3, l3, ii33)

        l2 = self.up2(pos2, l3, l2, ii32, ww32)
        l2 = self.dec2(pos2, l2, ii22)

        l1 = self.up1(pos1, l2, l1, ii21, ww21)
        l1 = self.dec1(pos1, l1, ii11)

        l0 = self.up0(pos0, l1, l0, ii10, ww10)
        l0 = self.dec0(pos0, l0, ii00)

        x = self.out_layer(l0)
        return F.normalize(x, p=2, dim=2)
