"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch.nn.functional as F
import torch
import torch.nn as nn
from performer_pytorch import SelfAttention
import math

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe


#######################################
#           ViT
######################################
def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, return_y=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_y:
            return y,x
        return x

class Transformer(nn.Module):
    def __init__(self,
                 in_dim,
                 num_emb,
                 nhead,
                 num_layers,
                 dropout,
                 pe='bin',
                 ):
        super(Transformer, self).__init__()

        self.emb = nn.Linear(in_dim, num_emb)
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            Block(
                dim=num_emb, num_heads=nhead)
            for _ in range(num_layers)])
        self.norm = nn.LayerNorm(num_emb)


        self.dropout = nn.Dropout(dropout)
        self.pe = pe

        if self.pe == 'lap':
            self.pe_enc = nn.Linear(10, num_emb)
        elif self.pe == 'mlp':
            self.pe_enc = nn.Linear(2, num_emb)
        elif self.pe == 'bin':
            self.pe_enc = nn.Embedding(11000, num_emb)
        elif self.pe == 'sinu':
            self.pe_enc = nn.Embedding.from_pretrained(positionalencoding2d(num_emb, 110, 100).flatten(1).T)

    def prepare_tokens(self, p, inputs):
        if self.pe == 'lap':
            pe_input = p.ndata['eigvec'] * (
                    torch.randint(0, 2, (p.ndata['eigvec'].shape[1],), dtype=torch.float, device=inputs.device)[
                    None, :] * 2 - 1)
        elif self.pe == 'mlp':
            pe_input = p
        elif self.pe in ['bin', 'sinu']:
            x = p[:, 0]
            y = p[:, 1]
            x = (x * 100).long()
            y = (y * 100).long()
            x[x >= 110] = 109
            y[y >= 100] = 99
            x[x < 0] = 0
            y[y < 0] = 0
            pe_input = x * 100 + y
        return self.dropout(self.emb(inputs) + self.pe_enc(pe_input))

    def forward(self, p, inputs):
        x = self.prepare_tokens(p,inputs)
        x = x.unsqueeze(0)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.squeeze()

    def get_last_selfattention(self, p, inputs):
        x = self.prepare_tokens(p, inputs)
        x = x.unsqueeze(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_last_y(self, p, inputs):
        x = self.prepare_tokens(p, inputs)
        x = x.unsqueeze(0)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_y=True)


class Performer(nn.Module):
    def __init__(self,
                 in_dim,
                 num_emb,
                 out_dim,
                 nhead,
                 num_layers,
                 dropout,
                 pe='bin',
                 ):
        super(Performer, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.pe = pe
        self.norm0 = nn.GroupNorm(nhead, num_emb)
        self.num_emd = num_emb
        self.act=nn.ReLU()

        self.attlayers = nn.ModuleList()
        self.fflayers = nn.ModuleList()
        self.norm1 = nn.ModuleList()
        self.norm2 = nn.ModuleList()

        for i in range(num_layers):
            self.attlayers.append(
                SelfAttention(
                    dim=num_emb, heads=nhead,
                    dropout=dropout, causal=False)
            )
            self.fflayers.append(nn.Sequential(
                nn.Linear(num_emb, num_emb * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(num_emb * 4, num_emb),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
            self.norm1.append(nn.GroupNorm(nhead, num_emb))
            self.norm2.append(nn.GroupNorm(nhead, num_emb))


        self.head = nn.Identity() if num_emb == out_dim else nn.Linear(num_emb, out_dim)
        self.emb = nn.Linear(in_dim, num_emb)

        if self.pe == 'lap':
            self.pe_enc = nn.Linear(10, num_emb)
        elif self.pe == 'mlp':
            self.pe_enc = nn.Linear(2, num_emb)
        elif self.pe == 'bin':
            self.pe_enc = nn.Embedding(11000, num_emb)
        elif self.pe == 'sinu':
            self.pe_enc = nn.Embedding.from_pretrained(positionalencoding2d(num_emb, 110, 100).flatten(1).T)


    def prepare_tokens(self, p, inputs):
        if self.pe == 'lap':
            pe_input = p.ndata['eigvec'] * (
                    torch.randint(0, 2, (p.ndata['eigvec'].shape[1],), dtype=torch.float, device=inputs.device)[
                    None, :] * 2 - 1)
        elif self.pe == 'mlp':
            pe_input = p
        elif self.pe in ['bin', 'sinu']:
            x = p[:, 0]
            y = p[:, 1]
            x = (x * 100).long()
            y = (y * 100).long()
            x[x >= 110] = 109
            y[y >= 100] = 99
            x[x < 0] = 0
            y[y < 0] = 0
            pe_input = x * 100 + y

        return self.dropout(self.emb(inputs) + self.pe_enc(pe_input))

    def forward(self, p, inputs):
        x = self.prepare_tokens(p,inputs)
        h = self.norm0(x)
        for l in range(self.num_layers):
            h = self.norm1[l](h + self.dropout(self.attlayers[l](h.unsqueeze(0))).squeeze(0))
            h = self.norm2[l](h + self.fflayers[l](h))
        return self.act(self.head(h))

    def get_last_selfattention(self, p, inputs):
        x = self.prepare_tokens(p, inputs)
        h = self.norm0(x)
        for l in range(self.num_layers-1):
            h = self.norm1[l](h + self.dropout(self.attlayers[l](h.unsqueeze(0))).squeeze(0))
            h = self.norm2[l](h + self.fflayers[l](h))
        h = self.attlayers[l](h.unsqueeze(0)).squeeze(0)
        return h


class SpaFormerModel(nn.Module):
    def __init__(
            self,
            backbone,
            in_dim: int,
            num_hidden: int,
            num_layers: int,
            nhead: int,
            feat_drop: float,
            mask_cell_rate: float = 0.2,
            mask_feature_rate: float = 0.3,
            latent_dim: int = 128,
            pe='bin',
            objective: str = 'mask',
    ):
        super(SpaFormerModel, self).__init__()
        self._mask_cell_rate = mask_cell_rate
        self._mask_feature_rate = mask_feature_rate
        self._objective = objective
        self.backbone = backbone['backbone']
        # build encoder
        # self.encoder = Performer(
        #     in_dim=in_dim,
        #     num_emb=num_hidden,
        #     out_dim=latent_dim,
        #     nhead=nhead,
        #     num_layers=num_layers,
        #     dropout=feat_drop,
        #     pe=pe,
        # )
        self.encoder = Transformer(
            in_dim=in_dim,
            num_emb=num_hidden,
            nhead=nhead,
            num_layers=num_layers,
            dropout=feat_drop,
            pe=pe,
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, num_hidden),
            nn.PReLU(),
            nn.Dropout(feat_drop),
            nn.Linear(num_hidden, in_dim),
            nn.ReLU()
        )

        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.encoder_to_decoder = nn.Identity()
        self.criterion = nn.MSELoss()

    def encoding_mask_noise(self, x):
        num_cells = x.shape[0]
        perm = torch.randperm(num_cells, device=x.device)

        # random masking
        num_masked_cells = int(self._mask_cell_rate * num_cells)

        masked_cells = perm[: num_masked_cells]
        kept_cells = perm[num_masked_cells:]

        # randomly dropout cells in the whole cell list
        in_x = torch.zeros(x.shape, device=x.device)
        in_x[kept_cells] = x[kept_cells]

        in_x = F.dropout(in_x, p=self._mask_feature_rate)
        masked_id = torch.zeros((x.shape[0],), device=x.device)
        masked_id[masked_cells] = 1

        return in_x

    def forward(self, p, images):
        x = self.backbone(images)
        print(x.shape)
        print(p.shape)
        test = input()

        masked_input_x = self.encoding_mask_noise(x)
        enc_rep = self.encoder(p, masked_input_x)
        rep = self.encoder_to_decoder(enc_rep)
        recon = self.decoder(rep)
        mse = nn.MSELoss()
        loss = mse(recon, x)
        return x,loss



class ContrastiveModel(nn.Module):
    def __init__(self, backbone, head='mlp', features_dim=128):
        super(ContrastiveModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.backbone_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.backbone_dim, self.backbone_dim),
                    nn.ReLU(), nn.Linear(self.backbone_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        backbone_feature = self.backbone(x)
        features = self.contrastive_head(backbone_feature)
        features = F.normalize(features, dim = 1)
        return features


class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1):
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters) for _ in range(self.nheads)])

    # def forward(self, x, size, forward_pass='default'):
    #     if forward_pass == 'default':
    #         features = self.backbone(x,size)
    #         out = [cluster_head(features) for cluster_head in self.cluster_head]
    #
    #     elif forward_pass == 'backbone':
    #         out = self.backbone(x,size)
    #
    #     elif forward_pass == 'head':
    #         out = [cluster_head(x) for cluster_head in self.cluster_head]
    #
    #     elif forward_pass == 'return_all':
    #         features = self.backbone(x,size)
    #         out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
    #
    #     else:
    #         raise ValueError('Invalid forward pass {}'.format(forward_pass))
    #
    #     return out

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}

        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))

        return out


