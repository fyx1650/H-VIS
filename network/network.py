import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_encoder(bbox, encoding, b_order=int(4), input_dim=1, num_levels=10, level_dim=2, init_level=3,
                base_resolution=4, desired_resolution=256, log2_hashmap_size=19, grid_type='hash'):
    if encoding == 'None':
        return lambda x: x, input_dim

    elif encoding == 'B_grid':
        from tpb_encoder import tpB_encoder
        encoder = tpB_encoder(bbox=bbox, input_dim=input_dim, num_levels=num_levels,
                              level_dim=level_dim, init_level=init_level,
                              base_resolution=base_resolution,
                              log2_hashmap_size=log2_hashmap_size,
                              desired_resolution=desired_resolution, grid_type=grid_type, order=b_order)

    else:
        raise NotImplementedError(
            'Unknown encoding mode, choose from [None, frequency, hashgrid, tiledgrid, densegrid, B_grid]')

    return encoder, encoder.output_dim


class SDFNetwork(nn.Module):
    def __init__(self,
                 # --Encoder-- #
                 bbox, encoding="None", b_order=int(4),
                 input_dim=1, num_levels=10, level_dim=2, init_level=3, base_resolution=4,
                 desired_resolution=256, log2_hashmap_size=19, grid_type='hash',
                 # --MLP-- #
                 num_layers=3, hidden_dim=64, skips=None, clip_sdf=None, geometric_init=False, geo_radius_init=0.0,
                 weight_norm=False, activation='softplus',  # choose from [relu, softplus, sine]
                 need_bias=True
                 ):
        super().__init__()

        if skips is None:
            skips = []
        self.in_dim = input_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.skips = skips
        self.clip_sdf = clip_sdf
        self.geometric_init = geometric_init
        self.geo_radius_init = geo_radius_init
        self.weight_norm = weight_norm
        self.activation = activation

        self.encoder, self.in_dim = get_encoder(bbox, encoding, b_order=b_order, input_dim=input_dim,
                                                num_levels=num_levels, level_dim=level_dim, init_level=init_level,
                                                base_resolution=base_resolution,
                                                desired_resolution=desired_resolution,
                                                log2_hashmap_size=log2_hashmap_size, grid_type=grid_type)

        backbone = []

        for l in range(self.num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = self.hidden_dim

            if l == self.num_layers - 1:
                out_dim = 1
            elif l + 1 in self.skips:
                out_dim = self.hidden_dim - in_dim
            else:
                out_dim = self.hidden_dim
            lin = nn.Linear(in_dim, out_dim, bias=need_bias)

            # if l == self.num_layers - 1:
            #     torch.nn.init.constant_(lin.bias, -geo_radius_init)

            # geometric initialization:https://github.com/amosgropp/IGR
            if self.geometric_init:
                if l == self.num_layers-1:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=1e-4)
                    if lin.bias is not None:
                        torch.nn.init.constant_(lin.bias, -self.geo_radius_init)
                else:
                    torch.nn.init.normal_(lin.weight, mean=0.0, std=np.sqrt(2) / np.sqrt(out_dim))
                    if lin.bias is not None:
                        torch.nn.init.constant_(lin.bias, 0.0)

            if self.weight_norm:
                lin = nn.utils.weight_norm(lin)
            backbone.append(lin)

        # self.first_bias = nn.Parameter(torch.empty(self.in_dim))
        # nn.init.constant_(self.first_bias, 0.0)
        self.backbone = nn.ModuleList(backbone)

    def forward(self, x):
        # x: [B, 3]

        h = self.encoder(x)
        # h = torch.sin(torch.pi * (h + self.first_bias) / 2)
        # if isinstance(self.encoder, nn.Module):
        #     h = torch.cat((h, x), dim=-1)
        for l in range(self.num_layers):
            if l in self.skips:
                h = torch.cat((h, x), dim=-1) / np.sqrt(2)
            h = self.backbone[l](h)
            if l < self.num_layers - 1:
                if self.activation == 'softplus':
                    h = nn.Softplus(beta=100)(h)
                elif self.activation == 'relu':
                    h = F.relu(h)
                elif self.activation == 'tanh':
                    h = nn.Tanh()(h)
                elif self.activation == 'sine':
                    h = torch.sin(torch.pi * h / 2)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h