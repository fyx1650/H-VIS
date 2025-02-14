import os
import argparse
import json
import numpy as np
import trimesh
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm
from utils import seed_everything
from trainer import Trainer
from dataset.Dataset_3d import SDFDataset_3D
from network import SDFNetwork, SIREN

with open("setup.json") as load_en:
    net_para = json.load(load_en)
load_en.close()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1]))


"""
This file only works for those dataset with testset.txt which record the ply files it contains, like ABC, Thingi10k, Famous,
or you can create such txt file for you own dataset.
"""


class MultiEpochsDataLoader(data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0, help="which gpu am I")

    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--workspace', type=str, default='workspace')

    parser.add_argument('--encoding', type=str, default='B_grid', help='choose from [None, B_grid]')
    parser.add_argument('--use_siren', default=False, action='store_true', help='whether to use siren net as model')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        help='choose from[Adam, Adamw, RMsprops, LBFGS]')
    parser.add_argument('--lr_type', type=str, default='cos', help='choose from [remain, pow, cos, anneal_cos]')

    parser.add_argument('--loss_func', type=str, default='l1', help='choose from [l1,mse]')
    parser.add_argument('--minsurf', default=False, action='store_true', help='whether to add minsurf term')
    parser.add_argument('--eikonal', default=False, action='store_true', help='whether to add eikonal term')
    parser.add_argument('--second_order', type=str, nargs='*', default=['Non'],
                        help='choose from [None, vis_hessian, laplacian, ddiv, hvp, morse, project]')

    parser.add_argument('--online_sampling', default=False, action='store_true',
                        help='whether to sample points online when training')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--test', action='store_true', help="test mode")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.local_rank}' if torch.cuda.is_available() else 'cpu')
            # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # ----Basic Parameters---- #
    seed = net_para["BasicInfo"]["seed"]
    max_epochs = net_para["BasicInfo"]["max_epochs"]
    init_lr = net_para["BasicInfo"]["init_lr"]
    num_samples = net_para["BasicInfo"][
        "num_samples"]  # if the input data is PointCloud ,this value would be set as its quantity
    batch_size = net_para["BasicInfo"]["batch_size"]
    batch_num = net_para["BasicInfo"]["batch_num"]
    input_dim = net_para["BasicInfo"]["input_dim"]
    dimension = str(input_dim) + 'd'
    # ----Encoder Parameters---- #
    b_order = int(net_para["TPBEncoder"]["b_order"])
    grid_type = net_para["TPBEncoder"]["grid_type"][0]
    num_levels = net_para["TPBEncoder"]["num_levels"]
    level_dim = net_para["TPBEncoder"]["level_dim"]
    init_level = net_para["TPBEncoder"]["init_level"]
    base_resolution = net_para["TPBEncoder"]["base_resolution"]
    desired_resolution = net_para["TPBEncoder"]["desired_resolution"]
    log2_hashmap_size = net_para["TPBEncoder"]["log2_hashmap_size"]
    # ----MLP Parameters ---- #
    hidden_dim = net_para["MLP"]["hidden_dim"]
    num_layers = net_para["MLP"]["num_layers"]
    need_bias = net_para["MLP"]["need_bias"]
    activation = net_para["MLP"]["activation"]
    skips = net_para["MLP"]["skips"]
    geometric_init = net_para["MLP"]["geometric_init"]
    geo_radius_init = net_para["MLP"]["geo_radius_init"]
    clip_sdf = net_para["MLP"]["clip_sdf"]
    weight_norm = net_para["MLP"]["weight_norm"]

    # seed_everything(seed)
    load_path = opt.dataset_path
    filelists = []
    with open(load_path + '/testset.txt', 'r') as f:
        lines = f.readlines()
        for line in lines[:-1]:
            filelists.append(line[:-1])  # remove '/n'
        filelists.append(lines[-1])

    for file in tqdm(filelists, desc="Preprocessing--Generate ply files with 1e5 points:"):
        mesh_path = load_path + '/03_meshes/' + file + '.ply'
        save_path = load_path + '/04_pts_ply_test/' + file + '.ply'
        mesh_gt = trimesh.load(mesh_path, force='mesh')
        pts = mesh_gt.sample(10 ** 5)
        mesh = trimesh.Trimesh(vertices=pts)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mesh.export(save_path)

    filelists_1 = [
                   '00014228_fb174aafb16d47abab609285_trimesh_002',
                   '00014489_f4297f01e3434034b7051ebb_trimesh_004',
                   '00017846_08893609d30e453493c4c079_trimesh_021',
                   '00991623_957e42412f4b1c9ec00e7db1_trimesh_003',
                   '00992690_ed0f9f06ad21b92e7ffab606_trimesh_002',
                   '00993706_f8bc5c196ab9685d0182bbed_trimesh_001',
                   '00993917_4049b13b8ff84e59b2cfc43a_trimesh_000'
                   ]
    filelists_2 = [
                   '00016002_4290cac5423e4bd1a5765333_trimesh_001',
                   '00992087_adbf0b351ea40b651859747a_trimesh_066',
                   'flower',
                   'serapis',
                   'LibertyBase'
                   ]
    filelists_3 = [
                   'Liberty',
                   'armadillo'
                   ]
    filelists_4 = ['00017012_cd8dbafbc2a3422eb55090d7_trimesh_000']

    if opt.loss_func == 'l1':
        criterion = torch.nn.L1Loss()
    elif opt.loss_func == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('unexpected loss function')

    for file in filelists:
        seed_everything(seed)
        mesh_path = load_path + '/03_meshes/' + file + '.ply'
        pts_path = load_path + '/04_pts_ply_1e5/' + file + '.ply'

        if file in filelists_1:
            net_para["Loss"]["coef_ms"] = 4.8
        if file in filelists_2:
            net_para["Loss"]["coef_mnfld"] = 150
            net_para["Loss"]["coef_ms"] = 2.5
        if file in filelists_3:
            net_para["Loss"]["coef_ms"] = 4.5
        if file in filelists_3:
            net_para["Loss"]["coef_ms"] = 4.9

        train_dataset = SDFDataset_3D(device, pts_path, mesh_path, size=batch_num, num_samples=num_samples, batchsize=batch_size,
                                      online=opt.online_sampling,
                                      model_scale=net_para["TPBEncoder"]["model_scale"],
                                      space_scale=net_para["TPBEncoder"]["space_scale"])
        # train_loader = data.DataLoader(train_dataset, batch_size=1, shuffle=True)
        train_loader = MultiEpochsDataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

        valid_dataset = SDFDataset_3D(device, pts_path, mesh_path, size=1, num_samples=num_samples, batchsize=batch_size,
                                      online=opt.online_sampling,
                                      model_scale=net_para["TPBEncoder"]["model_scale"],
                                      space_scale=net_para["TPBEncoder"]["space_scale"])
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

        if opt.use_siren:
            model = SIREN(in_dim=net_para["BasicInfo"]["input_dim"], init_type=net_para["SIREN"]["init_type"],
                          sphere_init_params=net_para["SIREN"]["sphere_init_params"])
        else:
            model = SDFNetwork(bbox=train_dataset.bbox, encoding=opt.encoding, b_order=b_order,
                               input_dim=input_dim, num_levels=num_levels, level_dim=level_dim,
                               init_level=init_level, base_resolution=base_resolution,
                               desired_resolution=desired_resolution, log2_hashmap_size=log2_hashmap_size,
                               grid_type=grid_type,
                               num_layers=num_layers, hidden_dim=hidden_dim, skips=skips, clip_sdf=clip_sdf,
                               geometric_init=geometric_init, geo_radius_init=geo_radius_init,
                               weight_norm=weight_norm, activation=activation, need_bias=need_bias)
        print(model)

        optimizer = lambda model: torch.optim.Adam([
                                                       {'name': 'encoding', 'params': filter(lambda p: p.requires_grad,
                                                                                             model.encoder.parameters())},
                                                       {'name': 'net', 'params': filter(lambda p: p.requires_grad,
                                                                                        model.backbone.parameters()),
                                                        'weight_decay': 0},
                                                   ] if isinstance(model.encoder, nn.Module) else [
            {'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()),
             'weight_decay': 0}],
                                                   lr=init_lr, betas=(0.9, 0.99), eps=1e-15)

        workspace = 'test_on_dataset/' + opt.dataset_name + '/SDF_' + file

        trainer = Trainer('ngp', model, device=device, net_para=net_para, workspace=workspace,
                          optimizer=optimizer, criterion=criterion, ema_decay=None,
                          fp16=opt.fp16, lr_type=opt.lr_type, use_siren=opt.use_siren,
                          use_checkpoint='latest', eval_interval=10,
                          minsurf=opt.minsurf, eikonal=opt.eikonal, second_order=opt.second_order,
                          dimension=dimension, max_epochs=max_epochs)
        trainer.train(train_loader, valid_loader, max_epochs)

        net_para["Loss"]["coef_mnfld"] = 200
        net_para["Loss"]["coef_ms"] = 4


