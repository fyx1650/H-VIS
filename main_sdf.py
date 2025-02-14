import os
import argparse
import json
import torch
import torch.nn as nn
import torch.utils.data as data
from utils import seed_everything
from trainer import Trainer

with open("setup.json") as load_en:
    net_para = json.load(load_en)
load_en.close()

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, 1]))


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

    parser.add_argument('--pts_path', type=str, default=None)
    parser.add_argument('--mesh_path', type=str, default=None)
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
    parser.add_argument('--provide_sdf', default=False, action='store_true',
                        help='whether to provide ground truth sdf values')
    parser.add_argument('--fp16', action='store_true', help="use amp mixed precision training")
    parser.add_argument('--test', action='store_true', help="test mode")

    opt = parser.parse_args()
    print(opt)

    device = torch.device(f'cuda:{opt.local_rank}' if torch.cuda.is_available() else 'cpu')
    # device = torch.device("cpu")

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

    seed_everything(seed)

    if opt.loss_func == 'l1':
        criterion = torch.nn.L1Loss()
    elif opt.loss_func == 'mse':
        criterion = torch.nn.MSELoss()
    else:
        raise ValueError('unexpected loss function')

    if dimension == '1d':
        from dataset.Dataset_1d import SDFDataset_1D

        sample_type = 'uniform'

        train_dataset = SDFDataset_1D(size=batch_num, num_samples=num_samples, batchsize=batch_size,
                                      online=opt.online_sampling)
        train_loader = MultiEpochsDataLoader(train_dataset, batch_size=1, shuffle=False)

        valid_dataset = SDFDataset_1D(size=1, num_samples=num_samples, batchsize=batch_size,
                                      online=opt.online_sampling)
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=False)

        net_para["Loss"]["coef_mnfld"] = 10
        net_para["Loss"]["coef_ms"] = 0.5
        net_para["Loss"]["coef_eikonal"] = 1
        net_para["TPBEncoder"]["base_resolution"] = 4
        net_para["TPBEncoder"]["desired_resolution"] = 256
    elif dimension == '2d':
        from dataset.Dataset_2d import get2D_dataset

        # shape_type = 'circle'
        # shape_type = 'double_circle'
        # shape_type = 'square'
        # shape_type = 'L'
        # shape_type = 'L_thin'
        # shape_type = 'hexagon'
        shape_type = 'snowflake'
        # shape_type = 'random_sf'
        # shape_type = 'noisy_circle'
        # shape_type = 'noisy_square'
        # shape_type = 'noisy_L'
        # shape_type = 'noisy_hexagon'
        # shape_type = 'noisy_snowflake'
        line_sample_type = 'uniform'
        train_dataset = get2D_dataset(device, batch_num, num_samples, batch_size, opt.online_sampling,
                                      net_para["TPBEncoder"]["space_scale"],
                                      shape_type=shape_type,
                                      line_sample_type=line_sample_type,
                                      model_scale=net_para["TPBEncoder"]["model_scale"])
        train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=False)
        # train_loader = MultiEpochsDataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

        valid_dataset = get2D_dataset(device, 1, num_samples, batch_size, opt.online_sampling,
                                      net_para["TPBEncoder"]["space_scale"],
                                      shape_type=shape_type,
                                      line_sample_type=line_sample_type,
                                      model_scale=net_para["TPBEncoder"]["model_scale"])
        valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True)
        net_para["Loss"]["coef_mnfld"] = 20
        net_para["Loss"]["coef_ms"] = 0.5
        net_para["Loss"]["coef_eikonal"] = 1
        net_para["TPBEncoder"]["base_resolution"] = 4
        net_para["TPBEncoder"]["desired_resolution"] = 256
    elif dimension == '3d':
        from dataset.Dataset_3d import SDFDataset_3D, SDFDataset_3D_online

        if opt.online_sampling:
            # this is just for sdf regression task
            train_dataset = SDFDataset_3D_online(device, opt.pts_path, opt.mesh_path, size=batch_num, num_samples=num_samples,
                                                 batchsize=batch_size,
                                                 online=opt.online_sampling,
                                                 model_scale=net_para["TPBEncoder"]["model_scale"],
                                                 space_scale=net_para["TPBEncoder"]["space_scale"])
            train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

            valid_dataset = SDFDataset_3D_online(device, opt.pts_path, opt.mesh_path, size=1, num_samples=num_samples,
                                                 batchsize=batch_size,
                                                 online=opt.online_sampling,
                                                 model_scale=net_para["TPBEncoder"]["model_scale"],
                                                 space_scale=net_para["TPBEncoder"]["space_scale"])
            valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

            net_para["BasicInfo"]["init_lr"] = 1e-4
            net_para["Loss"]["coef_mnfld"] = 1
            net_para["Loss"]["coef_nonmnfld"] = 1
            net_para["Loss"]["coef_near"] = 1
        else:
            train_dataset = SDFDataset_3D(device, opt.pts_path, opt.mesh_path, size=batch_num, num_samples=num_samples, batchsize=batch_size,
                                          online=opt.online_sampling,
                                          model_scale=net_para["TPBEncoder"]["model_scale"],
                                          space_scale=net_para["TPBEncoder"]["space_scale"])
            train_loader = data.DataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)
            # train_loader = MultiEpochsDataLoader(train_dataset, batch_size=1, num_workers=4, shuffle=True)

            valid_dataset = SDFDataset_3D(device, opt.pts_path, opt.mesh_path, size=1, num_samples=num_samples, batchsize=batch_size,
                                          online=opt.online_sampling,
                                          model_scale=net_para["TPBEncoder"]["model_scale"],
                                          space_scale=net_para["TPBEncoder"]["space_scale"])
            valid_loader = data.DataLoader(valid_dataset, batch_size=1, shuffle=True)
    else:
        raise ValueError('unsupported question type')

    from network import SDFNetwork, SIREN
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

    if opt.optimizer == 'Adam':
        if opt.use_siren:
            optimizer = lambda model: torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=0.0)
        else:
            optimizer = lambda model: torch.optim.Adam([
                {'name': 'encoding', 'params': filter(lambda p: p.requires_grad, model.encoder.parameters())},
                {'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0},
            ] if isinstance(model.encoder, nn.Module) else [{'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0}],
              lr=init_lr, betas=(0.9, 0.99), eps=1e-15)
    elif opt.optimizer == 'Adamw':
        if opt.use_siren:
            optimizer = lambda model: torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=0.0)
        else:
            optimizer = lambda model: torch.optim.AdamW([
                 {'name': 'encoding', 'params': filter(lambda p: p.requires_grad, model.encoder.parameters())},
                 {'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0},
            ] if isinstance(model.encoder, nn.Module) else [{'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0}],
              lr=init_lr, betas=(0.9, 0.99), eps=1e-15)
    elif opt.optimizer == 'RMSprop':
        optimizer = lambda model: torch.optim.RMSprop([
                    {'name': 'encoding', 'params': filter(lambda p: p.requires_grad, model.encoder.parameters())},
                    {'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0},
                ] if isinstance(model.encoder, nn.Module) else [{'name': 'net', 'params': filter(lambda p: p.requires_grad, model.backbone.parameters()), 'weight_decay': 0}],
              lr=init_lr, momentum=0.9, alpha=0.9, eps=1e-8, centered=False)
    elif opt.optimizer == 'LBFGS':
        optimizer = lambda model: torch.optim.LBFGS(model.parameters(), lr=init_lr)
    else:
        raise ValueError('unexpected optimizer')

    if opt.use_siren:
        workspace = 'test1/' + opt.workspace + '/SIREN/' + net_para["SIREN"]["init_type"] + '_init/' + ('ms+' if opt.minsurf else '') + \
                    ('e_' if opt.eikonal else '') + '+'.join(i for i in opt.second_order) + \
                    '/' + opt.loss_func + '_sample_' + str(num_samples) + '_batch_' + \
                    str(batch_size) + '_' + str(batch_num) + ('_online' if opt.online_sampling else '_fix') + \
                    '_' + opt.optimizer + '_1' + ('_fp16' if opt.fp16 else '')
    else:
        if 'None' in opt.encoding:
            workspace = 'test1/' + opt.workspace + '/onlymlp/' + ('ms+' if opt.minsurf else '') + \
                        ('e_' if opt.eikonal else '') + '+'.join(i for i in opt.second_order) + \
                        '/' + opt.loss_func + '_sample_' + str(num_samples) + '_batch_' + \
                        str(batch_size) + '_' + str(batch_num) + ('_online' if opt.online_sampling else '_fix') + \
                        '_' + opt.optimizer + '_1'
        else:
            workspace = 'test1/' + opt.workspace + '/res' + str(base_resolution) + '_' + str(desired_resolution) + \
                        '_level_' + str(num_levels) + '_dim_' + str(level_dim) + '/spline' + str(b_order) + '/' + \
                        ('withoutmlp/' if num_layers == 1 else 'withmlp/') + \
                        ('ms+' if opt.minsurf else '') + \
                        ('e+' if opt.eikonal else '') + '+'.join(i for i in opt.second_order) + \
                        '/' + opt.loss_func + '_sample_' + str(num_samples) + '_batch_' + \
                        str(batch_size) + '_' + str(batch_num) + ('_online' if opt.online_sampling else '_fix') + \
                        '_' + opt.optimizer + '_1' + ('_fp16' if opt.fp16 else '') + \
                        ('_sp' if net_para["MLP"]["activation"] == 'softplus' else '_sine')

    trainer = Trainer('ngp', model, device=device, net_para=net_para, workspace=workspace,
                      optimizer=optimizer, criterion=criterion, ema_decay=None,
                      fp16=opt.fp16, lr_type=opt.lr_type, use_siren=opt.use_siren,
                      use_checkpoint='latest', eval_interval=2, provide_sdf=opt.provide_sdf,
                      minsurf=opt.minsurf, eikonal=opt.eikonal, second_order=opt.second_order,
                      dimension=dimension, max_epochs=max_epochs)
    trainer.train(train_loader, valid_loader, max_epochs)

    if opt.test and dimension == '3d':
        trainer.draw_implicit_function_3d(train_dataset.bbox,
                                          train_dataset.center,
                                          train_dataset.scale,
                                          save_path=os.path.join(opt.workspace, 'results', 'output.ply'),
                                          resolution=1024)


