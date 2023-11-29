import glob

import numpy as np
import openpack_toolkit as optk
import openpack_torch as optorch
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from vit_pytorch.vit import Attention, FeedForward, PreNorm
import  math

class FeedForwardEmbedding(nn.Module):
    def __init__(self, indim, outdim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(indim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, outdim),
        )

    def forward(self, x):
        return self.net(x)


class Embedding(nn.Module):
    def __init__(self, cfg, num_classes=11, concat=True):
        super().__init__()
        self.set_param(cfg)
        self.concat = concat

        self.imu_patch_embedding = nn.Linear(self.imu_input_dim, self.imu_embedding_dim)
        self.e4acc_patch_embedding = nn.Linear(self.e4acc_input_dim, self.e4acc_embedding_dim)
        self.bbox_patch_embedding = nn.Linear(self.bbox_input_dim, self.bbox_embedding_dim)
        self.keypoint_patch_embedding = nn.Linear(self.keypoint_input_dim, self.keypoint_embedding_dim)

        self.ht_patch_embedding = nn.Embedding(2, self.ht_embedding_dim)
        self.printer_patch_embedding = nn.Embedding(2, self.printer_embedding_dim)


    def set_param(self, cfg):
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)
        assert not (cfg.model.resnet and cfg.model.mbconv)

        self.use_substitute_image = cfg.model.use_substitute_image
        self.use_substitute_emb = cfg.model.use_substitute_emb
        self.add_defect_info = cfg.model.add_defect_info

        self.dim = cfg.model.dim
        self.imu_input_dim = cfg.dataset.stream.imu_dim * int(
            cfg.dataset.stream.frame_rate_imu * cfg.model.time_step_width / 1000
        )
        self.imu_embedding_dim = cfg.model.imu_dim
        self.keypoint_input_dim = cfg.dataset.stream.keypoint_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.keypoint_embedding_dim = cfg.model.keypoint_dim
        self.e4acc_input_dim = cfg.dataset.stream.e4acc_dim * int(
            cfg.dataset.stream.frame_rate_e4acc * cfg.model.time_step_width / 1000
        )
        self.e4acc_embedding_dim = cfg.model.e4acc_dim
        self.bbox_input_dim = cfg.dataset.stream.bbox_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.bbox_embedding_dim = cfg.model.bbox_dim
        self.ht_embedding_dim = cfg.model.ht_dim
        self.printer_embedding_dim = cfg.model.printer_dim

        if hasattr(cfg.model, "embedding_method"):
            self.embedding_method = cfg.model.embedding_method
        else:
            self.embedding_method = "linear"

        self.concat_dim = (
            self.imu_embedding_dim
            + self.keypoint_embedding_dim
            #+ self.e4acc_embedding_dim
            + self.bbox_embedding_dim
            + self.ht_embedding_dim
            + self.printer_embedding_dim
        )

    def forward(
        self,
        imu,
        keypoint,
        e4acc,
        bbox,
        ht,
        printer,
    ):
        b = imu.shape[0]
        t = imu.shape[1]
        x_list = []
        name_list = []

        imu = rearrange(imu, "b t f d -> b t (f d)")
        x_imu = self.imu_patch_embedding(imu)
        x_list.append(x_imu)
        name_list.append("imu")

        e4acc = rearrange(e4acc, "b t f d -> b t (f d)")
        x_e4acc = self.e4acc_patch_embedding(e4acc)
        x_list.append(x_e4acc)
        name_list.append("e4acc")

        bbox = rearrange(bbox, "b t f d -> b t (f d)")
        x_bbox = self.bbox_patch_embedding(bbox)
        x_list.append(x_bbox)
        name_list.append("bbox")

        keypoint = rearrange(keypoint, "b t f d n -> b t (f d n)")
        x_keypoint = self.keypoint_patch_embedding(keypoint)
        x_list.append(x_keypoint)
        name_list.append("keypoint")

        if self.ht_embedding_dim != 0:
            x_ht = self.ht_patch_embedding(ht)
            x_list.append(x_ht)
            name_list.append("ht")
        if self.printer_embedding_dim != 0:
            x_printer = self.printer_patch_embedding(printer)
            x_list.append(x_printer)
            name_list.append("printer")

        if self.concat:
            return torch.concat(x_list, dim=2)
        else:
            return x_list


class OpenPackBase(nn.Module):
    def __init__(self, cfg, num_classes=11, concat=True):
        super().__init__()
        self.embedding = Embedding(cfg, num_classes, concat)
        self.set_param(cfg)
        self.num_classes = num_classes

    def set_param(self, cfg):
        assert not (cfg.model.use_substitute_image and cfg.model.use_substitute_emb)
        assert not (cfg.model.resnet and cfg.model.use_cnn_feature)

        self.num_patches = cfg.model.num_patches
        self.depth = cfg.model.depth
        self.heads = cfg.model.heads
        self.mlp_dim = cfg.model.mlp_dim
        self.dim_head = cfg.model.dim_head
        self.use_pe = cfg.model.use_pe
        self.emb_dropout_p = cfg.model.emb_dropout
        self.dropout_p = cfg.model.dropout
        self.threshold_b = cfg.model.threshold_boundary

        if cfg.model.dim == -1:
            self.dim = self.embedding.concat_dim
        else:
            self.dim = cfg.model.dim

        self.use_substitute_image = cfg.model.use_substitute_image
        self.use_substitute_emb = cfg.model.use_substitute_emb
        self.add_defect_info = cfg.model.add_defect_info

        self.imu_input_dim = cfg.dataset.stream.imu_dim * int(
            cfg.dataset.stream.frame_rate_imu * cfg.model.time_step_width / 1000
        )
        self.imu_embedding_dim = cfg.model.imu_dim
        self.keypoint_input_dim = cfg.dataset.stream.keypoint_dim * int(
            cfg.dataset.stream.frame_rate_keypoint * cfg.model.time_step_width / 1000
        )
        self.keypoint_embedding_dim = cfg.model.keypoint_dim
        self.e4acc_input_dim = cfg.dataset.stream.e4acc_dim * int(
            cfg.dataset.stream.frame_rate_e4acc * cfg.model.time_step_width / 1000
        )
        self.e4acc_embedding_dim = cfg.model.e4acc_dim
        self.bbox_embedding_dim = cfg.model.bbox_dim
        self.ht_embedding_dim = cfg.model.ht_dim
        self.printer_embedding_dim = cfg.model.printer_dim


class ConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        # PreNorm(dim, DConvAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        # PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                        PreNorm(dim, ConvLayer(dim, conv_k, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        # for attn, ff, conv in self.layers:
        for attn, conv in self.layers:
            x = attn(x) + x
            # x = ff(x) + x
            x = conv(x) + x
        return x


class SE(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class ConvLayer(nn.Module):
    def __init__(self, dim, conv_k, dropout=0.0, expansion=4):
        super().__init__()
        # self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding="same")
        hidden_dim = int(dim * expansion)

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=conv_k, padding="same"),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(dim, dim, kernel_size=1, padding="same"),
                nn.Dropout(dropout),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_k, padding="same"),
                nn.GroupNorm(10, hidden_dim),
                nn.GELU(),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, padding="same"),
                nn.GroupNorm(10, dim),
            )

    def forward(self, x):
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        return x


class DConvAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        # Primer: Searching for Efficient Transformers for Language Modeling
        self.dconv_q = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)
        self.dconv_k = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)
        self.dconv_v = nn.Conv2d(heads, heads, (3, 1), stride=1, padding="same", groups=heads)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
        )

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q = self.dconv_q(q)
        k = self.dconv_k(k)
        v = self.dconv_v(v)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


#from conformer.encoder import ConformerBlock
from conformer import ConformerBlock

class DotProductAttention(nn.Module):
    '''缩短点积注意力'''
 
    def __init__(self, dropout = 0.1, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        
        self.dropout = nn.Dropout(dropout)
        self.attend = nn.Softmax(dim=-1)
 
        # queries的形状：(batch_size，查询的个数，d)
        # keys的形状：(batch_size，“键－值”对的个数，d)
        # values的形状：(batch_size，“键－值”对的个数，值的维度)
        
    def forward(self, queries, keys, values):
        d = queries.shape[-1]
        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        self.attention_weights = self.attend(scores)
        return torch.bmm(self.dropout(self.attention_weights), values)
    
def transpose_qkv(X,num_heads):
    """为了多注意⼒头的并⾏计算⽽变换形状"""
    # 输⼊X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，num_hiddens/num_heads)
    X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)
    X = X.permute(0, 2, 1, 3)
    return X.reshape(-1, X.shape[2], X.shape[3])

def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
    X = X.permute(0, 2, 1, 3)
    return X.reshape(X.shape[0], X.shape[1], -1)    

class MultiHeadCrossAttention(nn.Module):     # dim, heads=heads, dim_head=dim_head, dropout=dropout    dim,dim,dim,dim,heads,dropout
    """多头注意⼒"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadCrossAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
        self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
        self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
        self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)

    def forward(self, queries, keys, values):
        # queries，keys，values的形状: (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # 经过变换后，输出的queries，keys，values 的形状:(batch_size*num_heads，查询或者“键－值”对的个数，num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values)
        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)
    

class AfterNormMHSA(nn.Module):   
    def __init__(self,dim,fn,fn2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.fn2 = fn2

    def forward(self,motion,spatial,guidance,**kwargs):
        return self.norm(self.fn(guidance,motion,motion, **kwargs) + motion) , self.norm(self.fn2(guidance,spatial,spatial, **kwargs) + spatial)
    
class AfterNormMHCA(nn.Module):  
    def __init__(self,dim,fn,fn2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.fn2 = fn2

    def forward(self,x,y,**kwargs):
        return self.norm(self.fn(x,y,y, **kwargs) + x) , self.norm(self.fn2(y,x,x, **kwargs) + y)
    
class AfterNormFFN(nn.Module):  
    def __init__(self,dim,fn,fn2):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.fn2 = fn2

    def forward(self,x,y,**kwargs):
        return self.norm(self.fn(x, **kwargs) + x) , self.norm(self.fn2(y, **kwargs) + y)
    
    
class SelfDoubleCrossConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.silu = nn.SiLU()
        self.final = nn.Linear(dim*2 , dim)
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        AfterNormMHSA(dim, MultiHeadCrossAttention(dim,dim,dim,dim,heads,dropout),MultiHeadCrossAttention(dim,dim,dim,dim,heads,dropout)),
                        AfterNormMHCA(dim, MultiHeadCrossAttention(dim,dim,dim,dim,heads,dropout),MultiHeadCrossAttention(dim,dim,dim,dim,heads,dropout)),
                        AfterNormFFN(dim, FeedForward(dim, mlp_dim, dropout=dropout),FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, motion, spatial, guidance):   # x 引导 y     x = q , y = kv
        for mhsa, mhca, ffn in self.layers:
            # SGF sparse guided fusion 
            motion = self.silu(motion + guidance)
            spatial = self.silu(spatial + guidance)
            motion, spatial = mhsa(motion, spatial, guidance)
            # MSAF motion-spatial attention fusion
            motion, spatial = mhca(motion, spatial)
            motion, spatial = ffn(motion, spatial)
        return self.norm(motion + spatial)

class Sparse_Guided_CrossModal_Module(nn.Module):
    def __init__(self, dim_in, dim_imu, dim_kps, dim_bbox, dim_ht, dim_pr, dim_out, heads, dim_head, mlp_dim, dropout_p, ff_ratio=4):
        super(Sparse_Guided_CrossModal_Module, self).__init__()

        dim_ffn = ff_ratio * dim_out
        self.silu = nn.SiLU()
        # Layers
        self.imulayer = nn.Linear(dim_imu, dim_out)
        self.kpslayer = nn.Linear(dim_kps, dim_out)
        self.bboxlayer = nn.Linear(dim_bbox, dim_out)
        self.htlayer = nn.Linear(dim_ht, dim_out)
        self.prlayer = nn.Linear(dim_pr, dim_out)
        
        self.cross = SelfDoubleCrossConvTransformer(dim_out, 4, heads, dim_head, mlp_dim, dropout_p)
        
    def forward(self, x_list):
        imu = x_list[0]
        kps = x_list[1]
        ht = x_list[2]
        pr = x_list[3]
        bbox = x_list[4]
        
        imu = self.imulayer(imu)
        kps = self.kpslayer(kps)
        bbox = self.bboxlayer(bbox)
        ht = self.htlayer(ht)
        pr = self.prlayer(pr)

        sparse_guidance = ht + pr
        motion_stream = imu
        spatial_stream = kps + bbox
        
        x = self.cross(motion_stream, spatial_stream, sparse_guidance)
        
        return x

class AfterNorm(nn.Module):   # 用于 MHCA
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self,x,y,**kwargs):
        return self.norm(self.fn(x,y,y, **kwargs) + x)
    
class AfterNorm2(nn.Module):   # 用于 MHCA
    def __init__(self,dim,fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        
    def forward(self,x,**kwargs):
        return self.norm(self.fn(x, **kwargs) + x)
    
class CrossConvTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, conv_k=3):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        AfterNorm(dim, MultiHeadCrossAttention(dim,dim,dim,dim,heads,dropout)),
#                         AfterNorm2(dim, ConvLayer(dim, conv_k, dropout=dropout)),
                        AfterNorm2(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )   
            
    def forward(self, x, y):   # x 引导 y     x = q , y = kv
        for attn, ffn in self.layers:
            z = attn(x,y)
            z = ffn(z)
        return z      


class Intermediate_Bottleneck(nn.Module):
    def __init__(self, dim_model, out_size):
        super(Intermediate_Bottleneck, self).__init__()

        # Inner Class Branch
        self.proj_1 = nn.Linear(dim_model * 2, out_size)
        self.proj_2 = nn.Linear(out_size, dim_model)
        self.lstm = nn.LSTM(
            dim_model, dim_model, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        
        self.conv_bound = nn.Conv1d(dim_model, 1, 1)
        
        # Inner Boundary Branch
        brb = [
            SingleStageTCN(1, dim_model, 1, 2) for _ in range(2)
        ]
        self.brb = nn.ModuleList(brb)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x):
        logits = self.proj_1(self.lstm(x)[0])
        x = x + self.proj_2(logits.softmax(dim=-1))
        
        out_bound = self.conv_bound(x.transpose(1,2))
        outputs_bound = [out_bound]
        for br_stage in self.brb:
            out_bound = br_stage(self.activation_brb(out_bound))
            outputs_bound.append(out_bound)
        
        return x, logits, torch.stack(outputs_bound).mean(0)


from typing import Any, Optional, Tuple

class MultiStageTCN(nn.Module):
    """
    Y. Abu Farha and J. Gall.
    MS-TCN: Multi-Stage Temporal Convolutional Network for Action Segmentation.
    In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2019

    parameters used in originl paper:
        n_features: 64
        n_stages: 4
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.stage1 = SingleStageTCN(in_channel, n_features, n_classes, n_layers)

        stages = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages - 1)
        ]
        self.stages = nn.ModuleList(stages)

        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # for training
            outputs = []
            out = self.stage1(x)
            outputs.append(out)
            for stage in self.stages:
                out = stage(self.activation(out))
                outputs.append(out)
            return outputs
        else:
            # for evaluation
            out = self.stage1(x)
            for stage in self.stages:
                out = stage(self.activation(out))
            return out


class SingleStageTCN(nn.Module):
    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_layers: int,
        **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.layers = nn.ModuleList(layers)
        self.conv_out = nn.Conv1d(n_features, n_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv_in(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation: int, in_channel: int, out_channels: int) -> None:
        super().__init__()
        self.conv_dilated = nn.Conv1d(
            in_channel, out_channels, 3, padding=dilation, dilation=dilation
        )
        self.conv_in = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv_dilated(x))
        out = self.conv_in(out)
        out = self.dropout(out)
        return x + out


class NormalizedReLU(nn.Module):
    """
    Normalized ReLU Activation prposed in the original TCN paper.
    the values are divided by the max computed per frame
    """

    def __init__(self, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(x)
        x /= x.max(dim=1, keepdim=True)[0] + self.eps

        return x


class EDTCN(nn.Module):
    """
    Encoder Decoder Temporal Convolutional Network
    """

    def __init__(
        self,
        in_channel: int,
        n_classes: int,
        kernel_size: int = 25,
        mid_channels: Tuple[int, int] = [128, 160],
        **kwargs: Any
    ) -> None:
        """
        Args:
            in_channel: int. the number of the channels of input feature
            n_classes: int. output classes
            kernel_size: int. 25 is proposed in the original paper
            mid_channels: list. the list of the number of the channels of the middle layer.
                        [96 + 32*1, 96 + 32*2] is proposed in the original paper
        Note that this implementation only supports n_layer=2
        """
        super().__init__()

        # encoder
        self.enc1 = nn.Conv1d(
            in_channel,
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout1 = nn.Dropout(0.3)
        self.relu1 = NormalizedReLU()

        self.enc2 = nn.Conv1d(
            mid_channels[0],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout2 = nn.Dropout(0.3)
        self.relu2 = NormalizedReLU()

        # decoder
        self.dec1 = nn.Conv1d(
            mid_channels[1],
            mid_channels[1],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout3 = nn.Dropout(0.3)
        self.relu3 = NormalizedReLU()

        self.dec2 = nn.Conv1d(
            mid_channels[1],
            mid_channels[0],
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout4 = nn.Dropout(0.3)
        self.relu4 = NormalizedReLU()

        self.conv_out = nn.Conv1d(mid_channels[0], n_classes, 1, bias=True)

        self.init_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 1
        x1 = self.relu1(self.dropout1(self.enc1(x)))
        t1 = x1.shape[2]
        x1 = F.max_pool1d(x1, 2)

        # encoder 2
        x2 = self.relu2(self.dropout2(self.enc2(x1)))
        t2 = x2.shape[2]
        x2 = F.max_pool1d(x2, 2)

        # decoder 1
        x3 = F.interpolate(x2, size=(t2,), mode="nearest")
        x3 = self.relu3(self.dropout3(self.dec1(x3)))

        # decoder 2
        x4 = F.interpolate(x3, size=(t1,), mode="nearest")
        x4 = self.relu4(self.dropout4(self.dec2(x4)))

        out = self.conv_out(x4)

        return out

    def init_weight(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


class ActionSegmentRefinementFramework(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel: 2048
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(
        self,
        in_channel: int,
        n_features: int,
        n_classes: int,
        n_stages: int,
        n_layers: int,
        n_stages_asb: Optional[int] = None,
        n_stages_brb: Optional[int] = None,
        **kwargs: Any
    ) -> None:

        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()
        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        shared_layers = [
            DilatedResidualLayer(2 ** i, n_features, n_features)
            for i in range(n_layers)
        ]
        self.shared_layers = nn.ModuleList(shared_layers)
        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)
        self.conv_bound = nn.Conv1d(n_features, 1, 1)

        # action segmentation branch
        asb = [
            SingleStageTCN(n_classes, n_features, n_classes, n_layers)
            for _ in range(n_stages_asb - 1)
        ]

        # boundary regression branch
        brb = [
            SingleStageTCN(1, n_features, 1, n_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_in(x)
        for layer in self.shared_layers:
            out = layer(out)

        out_cls = self.conv_cls(out)
        out_bound = self.conv_bound(out)

        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]

            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))
                outputs_bound.append(out_bound)

            return (outputs_cls, outputs_bound)
        else:
            for as_stage in self.asb:
                out_cls = as_stage(self.activation_asb(out_cls))

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound))

            return (out_cls, out_bound)

## boundary cross    refine
def unfold_1d(x, kernel_size=7, pad_value=0):
    B, C, T = x.size()
    padding = kernel_size // 2
    x = x.unsqueeze(-1)
    x = F.pad(x, (0, 0, padding, padding), value=pad_value)
    D = F.unfold(x, (kernel_size, 1), padding=(0, 0))
    return D.view(B, C, kernel_size, T)


def dual_barrier_weight(b, kernel_size=7, alpha=0.2):
    '''
    b: (B, 1, T)
    '''
    K = kernel_size
    b = unfold_1d(b, kernel_size=K, pad_value=20)
    # print('boundary unfild1d', b.shape)
    # b: (B, 1, K, T)
    HL = K // 2
    left = torch.flip(torch.cumsum(
        torch.flip(b[:, :, :HL + 1, :], [2]), dim=2), [2])[:, :, :-1, :]
    right = torch.cumsum(b[:, :, -HL - 1:, :], dim=2)[:, :, 1:, :]
    middle = torch.zeros_like(b[:, :, 0:1, :])
    # middle = b[:, :, HL:-HL, :]
    weight = alpha * torch.cat((left, middle, right), dim=2)
    return weight.neg().exp()


class LocalBarrierPooling(nn.Module):
    def __init__(self, kernel_size=1, alpha=0.2):
        super(LocalBarrierPooling, self).__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha

    def forward(self, x, barrier):
        '''
        x: (B, T, C)
        barrier: (B, ,T , 1) (>=0)
        '''

        xs = unfold_1d(x, self.kernel_size)
        # print('xs', xs.shape)
        w = dual_barrier_weight(barrier, self.kernel_size, self.alpha)

        return (xs * w).sum(dim=2) / ((w).sum(dim=2) + np.exp(-10))


class SigFormer(OpenPackBase):
    def __init__(
            self,
            num_classes,
            *args,
            **kargs,
    ):
        super().__init__(num_classes=num_classes, concat=False, *args, **kargs)
        self.depth = self.depth // 2

        self.dropout_imu = nn.Dropout(self.emb_dropout_p)
        self.layers_imu = ConvTransformer(
            self.imu_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_imu = Intermediate_Bottleneck(self.imu_embedding_dim, num_classes)

        self.dropout_keypoint = nn.Dropout(self.emb_dropout_p)
        self.layers_keypoint = ConvTransformer(
            self.keypoint_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_keypoint = Intermediate_Bottleneck(self.keypoint_embedding_dim, num_classes)

        self.dropout_bbox = nn.Dropout(self.emb_dropout_p)
        self.layers_bbox = ConvTransformer(
            self.bbox_embedding_dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.inner_module_bbox = Intermediate_Bottleneck(self.bbox_embedding_dim, num_classes)

        if self.use_pe:
            self.pos_embedding_imu = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.imu_embedding_dim)
            )
            self.pos_embedding_keypoint = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.keypoint_embedding_dim)
            )
            self.pos_embedding_bbox = nn.Parameter(
                torch.randn(1, self.num_patches + 1, self.bbox_embedding_dim)
            )

        self.fusion = Sparse_Guided_CrossModal_Module(self.dim, self.imu_embedding_dim, self.keypoint_embedding_dim,
                                                  self.bbox_embedding_dim, self.ht_embedding_dim,
                                                  self.printer_embedding_dim, self.dim, self.heads, self.dim_head,
                                                  self.mlp_dim, self.dropout_p)
        self.classlayers = ConvTransformer(
            self.dim, 3, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.ln = nn.LayerNorm(self.dim)
        self.lstm = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )
        self.linear_head = nn.Linear(self.dim * 2, num_classes)

        self.classlayers2 = ConvTransformer(
            self.dim, 2, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )

        self.classlayers3 = ConvTransformer(
            self.dim, 3, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )

        self.boundarylayers = ConvTransformer(
            self.dim, 2, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )
        self.boundarylayers2 = ConvTransformer(
            self.dim, 2, self.heads, self.dim_head, self.mlp_dim, self.dropout_p
        )

        self.cross1 = CrossConvTransformer(self.dim, 2, self.heads, self.dim_head, self.mlp_dim,
                                           self.dropout_p)  # self.dim ,2, self.heads, self.dim_head, self.mlp_dim, self.dropout_p

        self.cross2 = CrossConvTransformer(self.dim, 2, self.heads, self.dim_head, self.mlp_dim, self.dropout_p)

        self.lstm2 = nn.LSTM(
            self.dim, self.dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2
        )

        self.boundary_head = nn.Linear(self.dim * 2, 1)
        self.silu = nn.SiLU()
        self.finnal_layers = nn.Sequential(
            nn.Linear(
                num_classes,
                num_classes * 4,
            ),
            nn.SiLU(),
            nn.Linear(num_classes * 4, num_classes),
        )

        self.lbp_out = LocalBarrierPooling(99, alpha=0.1)
        self.sigmoid = nn.Sigmoid()


    def forward(
            self,
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
    ):
        t = imu.shape[1]
        x_list = self.embedding(
            imu,
            keypoint,
            e4acc,
            bbox,
            ht,
            printer,
        )
        assert len(x_list) == 6 and x_list[1].nelement() == 0

        x_imu = x_list[0]
        x_bbox = x_list[2]
        x_keypoint = x_list[3]
        x_ht = x_list[4]
        x_printer = x_list[5]
        inner_logit_list = []
        inner_boundary_list = []

        if self.use_pe:
            x_imu += self.pos_embedding_imu[:, :t]
        x_imu = self.dropout_imu(x_imu)
        x_imu = self.layers_imu(x_imu)
        x_imu, inner_logit, inner_boundary = self.inner_module_imu(x_imu)
        inner_logit_list.append(inner_logit)
        inner_boundary_list.append(inner_boundary)

        if self.use_pe:
            x_keypoint += self.pos_embedding_keypoint[:, :t]
        x_keypoint = self.dropout_keypoint(x_keypoint)
        x_keypoint = self.layers_keypoint(x_keypoint)
        x_keypoint, inner_logit, inner_boundary = self.inner_module_keypoint(x_keypoint)
        inner_logit_list.append(inner_logit)
        inner_boundary_list.append(inner_boundary)

        if self.use_pe:
           x_bbox += self.pos_embedding_bbox[:, :t]
        x_bbox = self.dropout_bbox(x_bbox)
        x_bbox = self.layers_bbox(x_bbox)
        x_bbox, inner_logit, inner_boundary = self.inner_module_bbox(x_bbox)
        inner_logit_list.append(inner_logit)
        inner_boundary_list.append(inner_boundary)

        x = self.fusion([x_imu, x_keypoint, x_ht, x_printer, x_bbox])
          
        feat = self.classlayers(x)

        # boundary branch
        bound = self.boundarylayers(x)
        bound = self.cross1(feat, bound)  # First-stage interaction
        bound = self.boundarylayers2(bound)

        boundary = self.ln(bound)
        boundary = self.lstm(boundary)[0]
        boundary = self.boundary_head(boundary)

        # class branch
        feat = self.classlayers2(feat)
        feat = self.cross2(bound, feat)  # Second-stage interaction
        feat = self.classlayers3(feat)

        feat = self.ln(feat)
        feat = self.lstm(feat)[0]
        feat = self.linear_head(feat)

        # filter
        barrier = torch.where(self.sigmoid(boundary) > self.threshold_b, self.sigmoid(boundary), 0.)
        # refine
        feat = self.lbp_out(feat, barrier)

        return feat, torch.stack(inner_logit_list).mean(0), boundary, torch.stack(inner_boundary_list).mean(0).transpose(1, 2)