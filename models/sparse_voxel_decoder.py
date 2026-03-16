import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmcv.cnn.bricks.transformer import FFN
from .sparsebev_transformer import SparseBEVSelfAttention, SparseBEVSampling, AdaptiveMixing
from .utils import DUMP, generate_grid, batch_indexing
from .bbox.utils import encode_bbox
import torch.nn.functional as F


def index2point(coords, pc_range, voxel_size):
    """
    coords: [B, N, 3], int
    pc_range: [-40, -40, -1.0, 40, 40, 5.4]
    voxel_size: float
    """
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device)
    return coords


def point2bbox(coords, box_size):
    """
    coords: [B, N, 3], float
    box_size: float
    """
    wlh = torch.ones_like(coords.float()) * box_size
    bboxes = torch.cat([coords, wlh], dim=-1)  # [B, N, 6]
    return bboxes


def upsample(pre_feat, pre_coords, interval):
    '''
    :param pre_feat: (Tensor), features from last level, (B, N, C)
    :param pre_coords: (Tensor), coordinates from last level, (B, N, 3) (3: x, y, z)
    :param interval: interval of voxels, interval = scale ** 2
    :param num: 1 -> 8
    :return: up_feat : upsampled features, (B, N*8, C//8)
    :return: up_coords: upsampled coordinates, (B, N*8, 3)
    '''
    pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    bs, num_query, num_channels = pre_feat.shape
    
    up_feat = pre_feat.reshape(bs, num_query, 8, num_channels // 8)  # [B, N, 8, C/8]
    up_coords = pre_coords.unsqueeze(2).repeat(1, 1, 8, 1).contiguous()  # [B, N, 8, 3]
    for i in range(len(pos_list)):
        up_coords[:, :, i + 1, pos_list[i]] += interval

    up_feat = up_feat.reshape(bs, -1, num_channels // 8)
    up_coords = up_coords.reshape(bs, -1, 3)

    return up_feat, up_coords


class SparseVoxelDecoder(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 semantic=False,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.semantic = semantic
        self.voxel_dim = [200, 200, 16]
        self.topk_training = topk_training
        self.topk_testing = topk_testing

        self.decoder_layers = nn.ModuleList()
        self.lift_feat_heads = nn.ModuleList()
        #self.occ_pred_heads = nn.ModuleList()
        
        if semantic:
            self.seg_pred_heads = nn.ModuleList()

        for i in range(num_layers):
            self.decoder_layers.append(SparseVoxelDecoderLayer(
                 embed_dims=embed_dims,
                 num_frames=num_frames,
                 num_points=num_points // (2 ** i),
                 num_groups=num_groups,
                 num_levels=num_levels,
                 pc_range=pc_range,
                 self_attn=i in [0, 1]
            ))
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))
            #self.occ_pred_heads.append(nn.Linear(embed_dims, 1))

            if semantic:
                self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()

    def forward(self, mlvl_feats, img_metas): # Corse-to-fine 稀疏采样解码
        # todo ------------------------------------------------#
        # todo coarse-to-fine structure 
        occ_preds = []
        
        topk = self.topk_training if self.training else self.topk_testing # train:[4000 16000 64000] test:[2000 8000 32000]

        B = len(img_metas)
        # init query coords
        interval = 2 ** self.num_layers # num_layers: 3 interval:8
        # ------------------------------------------------------#
        # interval：8 生成 (200/8 200/8 16/8)
        query_coord = generate_grid(self.voxel_dim, interval).expand(B, -1, -1)  # [B, N, 3] # generate_grid: 在3D空间中生成一个均匀分布的离散网格坐标
        # 初始的查询特征
        query_feat = torch.zeros([B, query_coord.shape[1], self.embed_dims], device=query_coord.device)  # [B, N, C]

        # ------------------------------------------------------#
        # 逐层细化
        for i, layer in enumerate(self.decoder_layers):
            DUMP.stage_count = i
            
            interval = 2 ** (self.num_layers - i)  # 8 4 2 1

            # todo --------------------------------------------------------------------#
            # todo 将离散的网格坐标变成真实世界的物理坐标，并根据当前缩放倍数编码
            # bbox from coords
            query_bbox = index2point(query_coord, self.pc_range, voxel_size=0.4)  # [B, N, 3] 网格索引 转换为 世界坐标
            query_bbox = point2bbox(query_bbox, box_size=0.4 * interval)  # [B, N, 6] 位置 -> 位置 + 宽长高(网格尺寸0.4)
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)  # [B, N, 6] # 编码
            # todo --------------------------------------------------------------------#
            # todo 空间特征提取：Transformer层，query从图像特征中采样
            # transformer layer
            query_feat = layer(query_feat, query_bbox, mlvl_feats, img_metas)  # [B, N, C]
            
            # upsample 2x 
            query_feat = self.lift_feat_heads[i](query_feat)  # [B, N, 8C] 将特征维度拉高
            query_feat_2x, query_coord_2x = upsample(query_feat, query_coord, interval // 2) # 每个大方块切分成8个小方块，空间分辨率提升一倍

            if self.semantic:
                seg_pred_2x = self.seg_pred_heads[i](query_feat_2x)  # [B, K, CLS]
            else:
                seg_pred_2x = None

            # todo --------------------------------------------------------------------#
            # todo 稀疏化筛选(剪枝)
            # sparsify after seg_pred
            non_free_prob = 1 - F.softmax(seg_pred_2x, dim=-1)[..., -1]  # [B, K] # 用以预测新切出来的小方块中，哪些非空
            indices = torch.topk(non_free_prob, k=topk[i], dim=1)[1]  # [B, K] # 只保留得分最高的N个位置
            #  注：topk筛选出的索引本身不可导。人为的，离散地丢弃不重要的点，下一层输入的query coord是一组全新的，被筛选过的采样点
            # todo --------------------------------------------------------------------#
            # 保留前topk非空个空间点：
            query_coord_2x = batch_indexing(query_coord_2x, indices, layout='channel_last')  # [B, K, 3] 
            query_feat_2x = batch_indexing(query_feat_2x, indices, layout='channel_last')  # [B, K, C]
            seg_pred_2x = batch_indexing(seg_pred_2x, indices, layout='channel_last')  # [B, K, CLS]

            occ_preds.append((
                torch.div(query_coord_2x, interval // 2, rounding_mode='trunc').long(),
                None,
                seg_pred_2x,
                query_feat_2x,
                interval // 2)
            )

            # todo ---------------------------------------#
            # todo .detach()
            query_coord = query_coord_2x.detach() # detach(): 让下一层仅根据上一层的结果细化，无需计算筛选梯度
            query_feat = query_feat_2x.detach()

        return occ_preds


class SparseVoxelDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True):
        super().__init__()

        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims), 
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        if self_attn:
            self.self_attn = SparseBEVSelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range, scale_adaptive=True)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None
        
        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )
        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)
        
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, query_bbox, mlvl_feats, img_metas):
        query_pos = self.position_encoder(query_bbox[..., :3])
        query_feat = query_feat + query_pos

        if self.self_attn is not None:
            query_feat = self.norm1(self.self_attn(query_bbox, query_feat))
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        query_feat = self.norm3(self.ffn(query_feat))

        return query_feat
