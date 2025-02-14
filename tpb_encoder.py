import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import time

from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

# ------------------------------tensor-product Bspline encoder(separate version)------------------------------ #


# change a decimal number to a base d number
def base_d_encoding(n, order, d):
    a = []
    while n >= order:
        a.append(n % order)
        n = n // order
    a.append(n)
    if len(a) < d:
        a = a + [0]*(d-len(a))
    # a.reverse()
    return a


def fast_hash(pos_grid):
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]
    result = torch.zeros_like(pos_grid)[:, 0]
    for i in range(pos_grid.shape[-1]):
        result ^= pos_grid[:, i] * primes[i]
    return result


def get_grid_index(grid_type, hashmap_size, resolution, order, pos_grid):
    index = 0
    stride = 1
    resolution = torch.squeeze(resolution, -1)
    hashmap_size = torch.squeeze(hashmap_size, -1)
    for i in range(pos_grid.shape[-1]):
        index += pos_grid[:, i] * stride
        stride *= (resolution + order - 1)
        # if stride > hashmap_size:
        #     stride = stride % hashmap_size
    if grid_type == 'hash':
        index = torch.where(stride > hashmap_size, fast_hash(pos_grid), index)
    return torch.remainder(index, hashmap_size)


def find_pos_index_tensor(pos, knots, order, bound=1):
    num = knots.shape[0]
    resolution = num - 2 * order + 1
    tmp = torch.ones_like(pos) * (num - order - 1)
    tmp2 = torch.ones_like(pos) * (order - 1)
    m1 = torch.floor((pos + bound) * resolution / (2 * bound)) + order - 1
    m2 = torch.where(m1 < order - 1, tmp2.long(), m1)
    index = torch.where(m2 > num - order - 1, tmp.long(), m2)
    return index


def cubic_bspline_api(pos_local):
    pos_local = pos_local
    return torch.stack(((1 - pos_local) * (1 - pos_local) * (1 - pos_local) / 6,
                        (3 * pos_local * pos_local * pos_local - 6 * pos_local * pos_local + 4) / 6,
                        (-3 * pos_local * pos_local * pos_local + 3 * pos_local * pos_local + 3 * pos_local + 1) / 6,
                        pos_local * pos_local * pos_local / 6), dim=-1)


def bspline_api(pos, knots, order, n=2):
    Vals = torch.zeros(pos.shape[0], pos.shape[1], order, device=pos.device)
    if n > 0:
        Ders = torch.zeros(n, pos.shape[0], pos.shape[1], order, device=pos.device)
    else:
        Ders = None
    with torch.no_grad():
        m = ((order-1) * torch.ones_like(pos)).long()
    Val = torch.zeros(pos.shape[0], pos.shape[1], order, order, device=pos.device)
    a = torch.zeros(pos.shape[0], pos.shape[1], 2, order, device=pos.device)
    Der = torch.zeros(pos.shape[0], pos.shape[1], n+1, order, device=pos.device)
    left = torch.zeros(pos.shape[0], pos.shape[1], order, device=pos.device)
    right = torch.zeros(pos.shape[0], pos.shape[1], order, device=pos.device)
    Val[:, :, 0, 0] = 1
    for k in range(1, order):
        left[:, :, k] = pos - knots[m + 1 - k]
        right[:, :, k] = knots[m + k] - pos
        saved = 0
        for r in range(k):
            Val[:, :, k, r] = right[:, :, r + 1] + left[:, :, k - r]
            temp = Val[:, :, r, k-1] / Val[:, :, k, r]
            Val[:, :, r, k] = saved + right[:, :, r + 1] * temp
            saved = left[:, :, k - r] * temp
        Val[:, :, k, k] = saved
    Vals += Val[:, :, :, order - 1]
    if Ders is not None:
        Der[:, :, 0, :] = Val[:, :, :, order-1]
        for k in range(order):
            s1 = 0
            s2 = 1
            a[:, :, 0, 0] = 1.0
            for c in range(1, n+1):
                d = 0
                rc = k-c
                pc = order-1-c
                if k >= c:
                    a[:, :, s2, 0] = a[:, :, s1, 0]/Val[:, :, pc+1, rc]
                    d = a[:, :, s2, 0] * Val[:, :, rc, pc]
                if rc >= -1:
                    j1 = 1
                else:
                    j1 = -rc
                if k - 1 <= pc:
                    j2 = c - 1
                else:
                    j2 = order - 1 - k
                for r in range(j1, j2 + 1):
                    a[:, :, s2, r] = (a[:, :, s1, r]-a[:, :, s1, r-1]) / Val[:, :, pc+1, rc+r]
                    d += a[:, :, s2, r] * Val[:, :, rc+r, pc]
                if k <= pc:
                    a[:, :, s2, c] = -a[:, :, s1, c-1]/Val[:, :, pc+1, k]
                    d += a[:, :, s2, c] * Val[:, :, k, pc]
                Der[:, :, c, k] = d
                t = s1
                s1 = s2
                s2 = t
        r = order - 1
        for k in range(n):
            Ders[k] = Der[:, :, k + 1, :] * r
            r *= (order - 1) - (k + 1)
    return Vals, Ders


# compute the value of bspline base function under some point functional
class bspline_base(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, inputs, knots, order, n=2):
        # inputs:[B, D], Vals:[B, D, order], Ders:[n, B, D, order]
        Vals, Ders = bspline_api(inputs, knots, order, n)
        ctx.save_for_backward(inputs, Ders)
        ctx.dims = n
        return Vals

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_outputs):
        inputs, Ders = ctx.saved_tensors
        n = ctx.dims
        if n == 1:
            # grad_inputs:[B, D]
            # grad_outputs = grad[0]  # grad_outputs:[B, D, order]
            # dz/dx=dz/dy*dy/dx, [B, D, 1, order] * [B, D, order, 1]
            grad_inputs = (grad_outputs * Ders[0, :, :, :]).sum(dim=-1)
        elif n == 2:
            grad_inputs = bspline_base_backward.apply(grad_outputs, inputs, Ders)
        else:
            raise ValueError('unexpected continuity')

        return grad_inputs, None, None, None


# calculate the second derivative
class bspline_base_backward(Function):

    @staticmethod
    @custom_fwd
    def forward(ctx, grad_outputs, inputs, Ders):
        # grad_outputs:[B, D, order], grad_inputs:[B, D]
        # dz/dx=dz/dy*dy/dx, [B, D, 1, order] * [B, D, order, 1] => ([B, D, order].*[B, D, order]).sum(dim=-1)
        grad_inputs = (grad_outputs * Ders[0, :, :, :]).sum(dim=-1)
        ctx.save_for_backward(Ders, grad_outputs)
        return grad_inputs

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_grad_inputs):
        Ders, sav_grad_outputs = ctx.saved_tensors
        # grad_grad_inputs [B, D]
        # [B, D, 1, 1] * [B, D, 1, order] => [B, D, 1].*[B, D, order]
        grad_inputs_grad_outputs = grad_grad_inputs.unsqueeze(-1) * Ders[0, :, :, :]
        # [B, D] .* ([B, D, 1, order] * [B, D, order, 1]) => [B, D] .* ([B, D, order].*[B, D, order]).sum(dim=-1)
        grad_inputs_grad_inputs = grad_grad_inputs * (sav_grad_outputs * Ders[1, :, :, :]).sum(dim=-1)
        return grad_inputs_grad_outputs, grad_inputs_grad_inputs, None


bspline_base = bspline_base.apply


class tpB_encoder(nn.Module):
    def __init__(self, bbox, input_dim=3, num_levels=1, level_dim=2, init_level=3, base_resolution=128,
                 desired_resolution=128, per_level_scale=2, log2_hashmap_size=19, grid_type='hash', order=int(4)
                 ):
        super(tpB_encoder, self).__init__()

        # the finest resolution desired at the last level, if provided, override per_level_scale
        if desired_resolution is not None:
            if num_levels != 1:
                per_level_scale = np.exp2(np.log2(desired_resolution / base_resolution) / (num_levels - 1))
            else:
                per_level_scale = 1

        self.input_dim = input_dim  # coord dims
        self.num_levels = num_levels  # levels number
        self.start_level = 0
        self.init_active_level = min(init_level, self.num_levels - 1)
        self.active_level = self.init_active_level
        self.level_dim = level_dim  # channel number
        self.per_level_scale = per_level_scale  # ratio of adjacent levels
        self.log2_hashmap_size = log2_hashmap_size
        self.base_resolution = base_resolution
        self.desired_resolution = desired_resolution
        self.output_dim = num_levels * level_dim
        self.grid_type = grid_type
        self.order = order

        # allocate parameters
        offsets = []
        offset = 0
        self.max_params = 2 ** log2_hashmap_size
        resolution_0 = np.ceil(base_resolution * per_level_scale ** np.arange(num_levels))
        resolutions = np.where(resolution_0 > desired_resolution, desired_resolution, resolution_0)
        params_in_levels = (resolutions + order - 1) ** input_dim
        if grid_type != 'dense':
            params_in_levels = np.where(params_in_levels > self.max_params,
                                        self.max_params, params_in_levels)  # limit max number
        for i in range(num_levels):
            offsets.append(offset)
            offset += params_in_levels[i]
        offsets.append(offset)
        offsets_buffer = torch.from_numpy(np.array(offsets, dtype=np.int32))
        base_d_lists = []
        for i in range(order ** input_dim):
            base_d_lists.append(base_d_encoding(i, order, input_dim))
        base_d_lists_buffer = torch.as_tensor(base_d_lists, dtype=torch.int)
        resolutions_buffer = torch.from_numpy(np.array(resolutions, dtype=np.int32))
        knots = torch.arange(-self.order + 1, self.order + 1)
        bbox = torch.from_numpy(bbox)

        self.register_buffer('bbox', bbox)
        self.register_buffer('knots', knots)
        self.register_buffer('offsets', offsets_buffer)
        self.register_buffer('resolutions', resolutions_buffer)
        self.register_buffer('base_d_lists', base_d_lists_buffer)

        # parameters
        self.embeddings = nn.Parameter(torch.empty(int(offset), level_dim, dtype=torch.float32))
        self.reset_parameters()

    def __repr__(self):
        return f"TPBEncoder: input_dim={self.input_dim} num_levels={self.num_levels} level_dim={self.level_dim} " \
               f"resolution={self.base_resolution} -> " \
               f"{int(round(self.base_resolution * self.per_level_scale ** (self.num_levels - 1)))} " \
               f"per_level_scale={self.per_level_scale:.4f} params={tuple(self.embeddings[i].shape for i in range(self.num_levels-1))} " \
               f"grid_type={self.grid_type}"

    def get_coefficients_index(self, pos_grid, base_index, res, hashmap_size):
        coefficients_coords = pos_grid.repeat_interleave(self.base_d_lists.shape[0], dim=0) + base_index
        coefficients_index = get_grid_index(self.grid_type, hashmap_size, res, self.order, coefficients_coords)
        return coefficients_index

    def reset_parameters(self):
        std = 1e-4
        self.embeddings.data.uniform_(-std, std)

    def set_active_level(self, level):
        self.active_level = level

    def forward(self, inputs):
        # inputs: [batch size(B), input_dim(D)], normalized real world positions in [-bound, bound]
        # return: [batch size, num_levels(L) * level_dim(C)] or [batch size, level_dim(C)]

        prefix_shape = list(inputs.shape[:-1])
        inputs = inputs.view(-1, self.input_dim)
        B, D = inputs.shape
        L = self.num_levels
        boxrange = self.bbox[1] - self.bbox[0]

        res = self.resolutions.unsqueeze(1).repeat(inputs.shape[0], 1)
        pos_repeat = inputs.repeat_interleave(self.resolutions.shape[0], dim=0)
        with torch.no_grad():
            m1 = torch.floor((pos_repeat - self.bbox[0]) * res / boxrange)
            m2 = torch.where(m1 <= 0, 0.0, m1)
            pos_grid = torch.where(m2 >= res, res - 1, m2)

            base_index = self.base_d_lists.repeat(L * B, 1)
            hashmap_size = self.offsets[1:] - self.offsets[:-1]
            hashmap_size = hashmap_size.unsqueeze(1).repeat(inputs.shape[0], 1).\
                repeat_interleave(self.base_d_lists.shape[0], dim=0).int()  # [64 * L * B,]
            offset = self.offsets[:-1].unsqueeze(1).repeat(inputs.shape[0], 1).\
                repeat_interleave(self.base_d_lists.shape[0], dim=0).int()
            coefficients_index = self.get_coefficients_index(pos_grid.int(), base_index,
                                                             res.repeat_interleave(self.base_d_lists.shape[0], dim=0),
                                                             hashmap_size)  # [B * order ** D]
            coefficients_index.unsqueeze_(1)
            coefficients_index = coefficients_index.repeat(1, self.level_dim) + offset

        pos_local = (pos_repeat - self.bbox[0]) * res / boxrange - pos_grid  # [0, 1]
        # base_val = cubic_bspline_api(pos_local)  # base_val:[L * B, D, order]
        base_val = bspline_base(pos_local, self.knots, self.order, 2)
        # base_val = base_val * box_range.unsqueeze(-1).repeat_interleave(self.order, dim=-1) / 2
        func_val = torch.squeeze(torch.gather(base_val.repeat_interleave(self.base_d_lists.shape[0], dim=0), dim=-1,
                                              index=base_index.unsqueeze(-1).long()), -1).prod(dim=-1, keepdim=True)  # func_val:[order ** D * L * B, level_dim]
        outputs = func_val * torch.gather(self.embeddings, dim=0, index=coefficients_index.long())
        outputs = outputs.reshape(L * B, self.order ** D, self.level_dim).sum(dim=1)  # [L * B, level_dim]
        outputs = outputs.view(prefix_shape + [self.output_dim])  # [B, L * leve_dim]

        return outputs




