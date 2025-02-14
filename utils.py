import os
import glob
import random
import time
import math
import json

import torch
import mcubes
import pysdf
import trimesh
import tqdm
import tensorboardX

import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.backends.cudnn
import matplotlib.style
import plotly.io

import plotly.figure_factory as ff

from packaging.version import parse as pa
from torch.autograd import grad
from scipy.spatial import cKDTree
from scipy import interpolate
from rich.console import Console
from torch_ema import ExponentialMovingAverage
# from pytorch3d.loss import chamfer_distance

from matplotlib import pyplot as plt
from plotly import graph_objects as go
from plotly.subplots import make_subplots


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.benchmark = True


def gradient(inputs, outputs, create_graph=True, retain_graph=True):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=create_graph,
        # allow_unused=True,
        retain_graph=retain_graph,
        only_inputs=True)[0]
    return points_grad


def get_aver(distances, face):
    return (distances[face[0]] + distances[face[1]] + distances[face[2]]) / 3.0


def load_mesh_and_normalize(path):
    mesh = trimesh.load(path, force='mesh')
    if isinstance(mesh, trimesh.Trimesh):
        # ---- normalize to [-1, 1] ---- #
        vs = mesh.vertices
        v_center = mesh.centroid
        vs = vs - v_center
        v_scale = np.abs(np.asarray(vs)).max()
        vs = vs / v_scale
        mesh.vertices = vs
        return mesh, vs, mesh.faces, v_center, v_scale
    else:
        raise ValueError('please input mesh file')


def remove_far(gt_pts, mesh, dis_trunc=0.1, is_use_prj=False):
    # gt_pts: trimesh
    # mesh: trimesh

    gt_kd_tree = cKDTree(gt_pts)
    distances, vertex_ids = gt_kd_tree.query(mesh.vertices, p=2, distance_upper_bound=dis_trunc)
    faces_remaining = []
    faces = mesh.faces

    if is_use_prj:
        normals = gt_pts.vertex_normals
        closest_points = gt_pts.vertices[vertex_ids]
        closest_normals = normals[vertex_ids]
        direction_from_surface = mesh.vertices - closest_points
        distances = direction_from_surface * closest_normals
        distances = np.sum(distances, axis=1)

    for i in range(faces.shape[0]):
        if get_aver(distances, faces[i]) < dis_trunc:
            faces_remaining.append(faces[i])
    mesh_cleaned = mesh.copy()
    mesh_cleaned.faces = faces_remaining
    mesh_cleaned.remove_unreferenced_vertices()

    return mesh_cleaned


def remove_nans(array):
    array_nan = np.isnan(array[:, 2])
    return array[~array_nan, :]


def generate_npz(load_path, save_path, num_smaples=2 ** 24):
    mesh, vs, faces, center, scale = load_mesh_and_normalize(load_path)
    vs_mn = mesh.sample(2 ** 24)
    vs_near = vs_mn + 0.01 * np.random.randn(num_smaples, 3)
    vs_far = np.array([-1.1, -1.1, -1.1]) + np.array([2.2, 2.2, 2.2]) * np.random.rand(2 ** 24, 3)
    sdf_fn = pysdf.SDF(vs, faces)
    sdf_near = sdf_fn(vs_near)
    sdfs_far = sdf_fn(vs_far)
    p_near = np.concatenate((vs_near, sdf_near[:, None]), axis=-1)
    p_far = np.concatenate((vs_far, sdfs_far[:, None]), axis=-1)
    np.savez(save_path, mnfld=vs_mn, near=p_near, far=p_far)


# ----------------------------VISUALIZATION---------------------------- #

def compute_curvature(inputs, model, c_type='gaussian'):
    cur = []
    if c_type == 'gaussian':
        for p in torch.split(inputs, 5000):
            sdf = model(p)
            g = gradient(p, sdf)
            hess_1 = torch.zeros(p.shape[0], p.shape[1] + 1, p.shape[1] + 1, dtype=p.dtype, device=p.device)
            for i in range(g.shape[1]):
                hess_1[:, i, :-1] += gradient(p, g[:, i])
            hess_1[:, -1, :-1] += g
            hess_1[:, :-1, -1] += g
            g_c_temp_1 = -torch.det(hess_1).squeeze() / g.norm(2, dim=-1) ** 4
            cur.append(g_c_temp_1.detach().cpu().numpy())
    elif c_type == 'mean':
        for p in torch.split(inputs, 5000):
            sdf = model(p)
            g = gradient(p, sdf)
            g_n = g / g.norm(2, dim=-1, keepdim=True)
            lapla = torch.zeros(p.shape[0], p.shape[1], dtype=p.dtype, device=p.device)
            for i in range(g.shape[1]):
                lapla[:, i] = gradient(p, g_n[:, i])[:, i]
            g_c_temp = -lapla.sum(dim=-1) / 2
            cur.append(g_c_temp.detach().cpu().numpy())
    else:
        raise ValueError('unexpected curvature type')
    cur = np.concatenate(cur, axis=0)
    return cur


def map_curvature_to_color(cur, vertices):
    v_rgba = np.zeros((vertices.shape[0], vertices.shape[1] + 1))
    v_rgba[:, -1] = 1.0
    min_c = np.min(cur)
    max_c = np.max(cur)
    cur_knots = np.linspace(min_c, max_c, num=5)
    idx_0 = np.where(cur <= cur_knots[0])
    if len(idx_0[0]) != 0:
        v_rgba[idx_0[0], 0] = 0.0
        v_rgba[idx_0[0], 1] = 0.0
        v_rgba[idx_0[0], 2] = 1.0
    idx_1 = np.where((cur > cur_knots[0]) & (cur <= cur_knots[1]))
    if len(idx_1[0]) != 0:
        v_rgba[idx_1[0], 0] = 0.0
        v_rgba[idx_1[0], 1] = (cur[idx_1[0]] - cur_knots[0]) / (cur_knots[1] - cur_knots[0])
        v_rgba[idx_1[0], 2] = 1.0
    idx_2 = np.where((cur > cur_knots[1]) & (cur <= cur_knots[2]))
    if len(idx_2[0]) != 0:
        v_rgba[idx_2[0], 0] = 0.0
        v_rgba[idx_2[0], 1] = 1.0
        v_rgba[idx_2[0], 2] = 1.0 - (cur[idx_2[0]] - cur_knots[1]) / (cur_knots[2] - cur_knots[1])
    idx_3 = np.where((cur > cur_knots[2]) & (cur <= cur_knots[3]))
    if len(idx_3[0]) != 0:
        v_rgba[idx_3[0], 0] = (cur[idx_3[0]] - cur_knots[2]) / (cur_knots[3] - cur_knots[2])
        v_rgba[idx_3[0], 1] = 1.0
        v_rgba[idx_3[0], 2] = 0.0
    idx_4 = np.where((cur > cur_knots[3]) & (cur <= cur_knots[4]))
    if len(idx_0[0]) != 0:
        v_rgba[idx_4[0], 0] = 1.0
        v_rgba[idx_4[0], 1] = 1.0 - (cur[idx_4[0]] - cur_knots[3]) / (cur_knots[4] - cur_knots[3])
        v_rgba[idx_4[0], 2] = 0.0
    idx_5 = np.where(cur > cur_knots[4])
    if len(idx_5[0]) != 0:
        v_rgba[idx_5[0], 0] = 1.0
        v_rgba[idx_5[0], 1] = 0.0
        v_rgba[idx_5[0], 2] = 0.0
    v_color = (v_rgba * 255).astype(np.uint8)
    return v_color


def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pa(torch.__version__) < pa('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def extract_fields(bound_min, bound_max, res, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], res[0]).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], res[1]).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], res[2]).split(N)

    u = np.zeros(res.numpy(), dtype=np.float32)
    for xi, xs in enumerate(X):
        for yi, ys in enumerate(Y):
            for zi, zs in enumerate(Z):
                xx, yy, zz = custom_meshgrid(xs, ys, zs)
                pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)  # [N, 3]
                val = query_func(pts).reshape(len(xs), len(ys),
                                              len(zs)).detach().cpu().numpy()  # [N, 1] --> [x, y, z]
                u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val

    # u = u[:, resolution // 2:, :]
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    # print('threshold: {}'.format(threshold))
    boxrange = bound_max - bound_min
    shortest_cut = torch.min(boxrange) / resolution
    max_res = torch.tensor([resolution * 4])
    res_x = min(max_res, boxrange[0] // shortest_cut + 1).int()
    res_y = min(max_res, boxrange[1] // shortest_cut + 1).int()
    res_z = min(max_res, boxrange[2] // shortest_cut + 1).int()
    res = torch.tensor([res_x, res_y, res_z])
    u = extract_fields(bound_min, bound_max, res, query_func)

    # print(u.shape, u.max(), u.min(), np.percentile(u, 50))

    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()
    res_np = res.detach().cpu().numpy()

    vertices, triangles = mcubes.marching_cubes(u, threshold)
    vertices = vertices / (res_np - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


def plot_cuts_iso(decode_points_func,
                  bbox, max_n_eval_pts=1e6,
                  resolution=256, thres=0.0, imgs_per_cut=1, save_path=None, device=None) -> go.Figure:
    # ref: https://github.com/bearprin/Neural-Singular-Hessian/blob/master/utils/visualizations.py
    """ plot levelset at a certain cross section, assume inputs are centered
    Args:
        decode_points_func: A function to extract the SDF/occupancy logits of (N, 3) points
        box_size (List[float]): bounding box dimension
        max_n_eval_pts (int): max number of points to evaluate in one inference
        resolution (int): cross section resolution xy
        thres (float): levelset value
        imgs_per_cut (int): number of images for each cut (plotted in rows)
    Returns:
        a numpy array for the image
    """
    if device is None:
        device = torch.device(f'cuda:{1}' if torch.cuda.is_available() else 'cpu')

    # xmax, ymax, zmax = [b / 2 for b in box_size]
    # xx, yy = np.meshgrid(np.linspace(-xmax, xmax, resolution),
    #                      np.linspace(-ymax, ymax, resolution))
    xmin = bbox[0, 0]
    xmax = bbox[1, 0]
    ymin = bbox[0, 1]
    ymax = bbox[1, 1]
    zmin = bbox[0, 2]
    zmax = bbox[1, 2]
    xxz, xzz = np.meshgrid(np.linspace(xmin, xmax, resolution + 1),
                           np.linspace(zmin, zmax, resolution + 1))
    xxy, xyy = np.meshgrid(np.linspace(xmin, xmax, resolution + 1),
                           np.linspace(ymin, ymax, resolution + 1))
    yyz, yzz = np.meshgrid(np.linspace(ymin, ymax, resolution + 1),
                           np.linspace(zmin, zmax, resolution + 1))
    xxz = xxz.ravel()
    xzz = xzz.ravel()
    xxy = xxy.ravel()
    xyy = xyy.ravel()
    yyz = yyz.ravel()
    yzz = yzz.ravel()

    fig = make_subplots(rows=imgs_per_cut, cols=3,
                        subplot_titles=('xz', 'xy', 'yz'),
                        # shared_xaxes='all', shared_yaxes='all',
                        vertical_spacing=0.01, horizontal_spacing=0.01,
                        )

    def _plot_cut(fig_in, idx, pos, query_func, rmin, rmax, cmin, cmax, res, device_in=None):
        """ plot one cross section pos (3, N) """
        # evaluate points in serial
        field_input = torch.tensor(pos.T, dtype=torch.float).to(device_in)
        feat = torch.zeros((field_input.shape[0], 1)).to(device_in)
        feat[:, 0] = 1
        values = query_func(field_input).flatten()
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
            # np.save('xy.npy', values)
        values = values.reshape(res + 1, res + 1)
        contour_dict = dict(autocontour=False,
                            colorscale='RdBu',
                            reversescale=True,
                            colorbar=dict(tickfont=dict(size=60), thickness=60, tickwidth=3),
                            line=dict(width=3),
                            contours=dict(
                                start=-0.5 + thres,
                                end=0.5 + thres,
                                size=0.05,
                                showlabels=False,  # show labels on contours
                                # labelfont=dict(  # label font properties
                                #     size=12,
                                #     color='white',
                                # )
                            ), )
        r_idx = idx // 3

        fig_in.add_trace(
            go.Contour(x=np.linspace(rmin, rmax, res + 1),
                       y=np.linspace(cmin, cmax, res + 1),
                       z=values,
                       **contour_dict
                       ),
            col=idx % 3 + 1, row=r_idx + 1  # 1-idx
        )

        # zero line
        fig_in.add_trace(
            go.Contour(x=np.linspace(rmin, rmax, res + 1),
                       y=np.linspace(cmin, cmax, res + 1),
                       z=values,
                       contours=dict(start=0, end=0, coloring='lines'),
                       line=dict(width=10),
                       showscale=False,
                       colorscale=[[0, 'rgb(0, 0, 0)'],
                                   [1, 'rgb(0, 0, 0)']]),
            col=idx % 3 + 1, row=r_idx + 1  # 1-idx
        )

        fig_in.update_xaxes(
            scaleanchor='y',
            scaleratio=1,
            range=[rmin, rmax],  # sets the range of xaxis
            constrain="range",  # meanwhile compresses the xaxis by decreasing its "domain"
            col=idx % 3 + 1, row=r_idx + 1)
        fig_in.update_yaxes(
            scaleanchor='x',
            scaleratio=1,
            range=[cmin, cmax],
            col=idx % 3 + 1, row=r_idx + 1
        )

    steps = np.stack([np.linspace(xmin, xmax, imgs_per_cut + 2)[1:-1],
                      np.linspace(ymin, ymax, imgs_per_cut + 2)[1:-1],
                      np.linspace(zmin, zmax, imgs_per_cut + 2)[1:-1]], axis=-1)
    for index in range(imgs_per_cut):
        position_cut = [np.vstack([xxz, np.full(xxz.shape[0], steps[index, 1]), xzz]),
                        np.vstack([xxy, xyy, np.full(xxy.shape[0], steps[index, 2])]),
                        np.vstack([np.full(yyz.shape[0], steps[index, 0]), yyz, yzz]), ]
        _plot_cut(
            fig, index * 3, position_cut[0], decode_points_func, xmin, xmax, zmin, zmax, resolution, device)
        _plot_cut(
            fig, index * 3 + 1, position_cut[1], decode_points_func, xmin, xmax, ymin, ymax, resolution, device)
        _plot_cut(
            fig, index * 3 + 2, position_cut[2], decode_points_func, ymin, ymax, zmin, zmax, resolution, device)

    fig.update_layout(
        title='iso-surface',
        height=1200 * imgs_per_cut,
        width=1200 * 3,
        autosize=False,
        scene=dict(aspectratio=dict(x=1, y=1))
    )

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # fig.write_html(save_path)
        fig.write_image(save_path)

    return fig


# ----------------------------EVALUATION---------------------------- #

def distance_p2p(points_src, normals_src, points_tgt, normals_tgt):
    kdtree = cKDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    dist_l1, idx_l1 = kdtree.query(points_src, p=1)
    if normals_src is not None and normals_tgt is not None:
        normals_src = normals_src / (np.linalg.norm(normals_src, axis=-1, keepdims=True) + 1e-6)
        normals_tgt = normals_tgt / (np.linalg.norm(normals_tgt, axis=-1, keepdims=True) + 1e-6)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        normals_dot_product = np.abs(normals_dot_product)
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    return dist, dist_l1, normals_dot_product


# calculate L1 chamfer distance, L2 chamfer distance, normals correctness and f-score based on point clouds
def eval_pts(mesh_gt, mesh_pred, normals_gt_all=None, num_samples=10 ** 5, miu=5e-3):
    if isinstance(mesh_gt, trimesh.Trimesh):
        points_gt, idx_gt = mesh_gt.sample(num_samples, return_index=True)
        points_gt = points_gt.astype(np.float32)
        normals_gt = mesh_gt.face_normals[idx_gt]
    else:
        vs_gt = np.asarray(mesh_gt.vertices, dtype=np.float32)
        choose_array = np.random.choice(vs_gt.shape[0], num_samples, replace=False)
        points_gt = vs_gt[choose_array, :]
        if normals_gt_all is not None:
            normals_gt = normals_gt_all[choose_array, :]
        else:
            normals_gt = None
    points_pred, idx_pred = mesh_pred.sample(num_samples, return_index=True)
    points_pred = points_pred.astype(np.float32)
    normals_pred = mesh_pred.face_normals[idx_pred]

    completeness, completeness_l1, normals_completeness = distance_p2p(points_gt, normals_gt, points_pred, normals_pred)
    recall = np.count_nonzero(completeness < miu) / num_samples
    completeness_l2 = completeness ** 2
    completeness_l1 = completeness_l1.mean()
    completeness_l2 = completeness_l2.mean()
    normals_completeness = normals_completeness.mean()
    accuracy, accuracy_l1, normals_accuracy = distance_p2p(points_pred, normals_pred, points_gt, normals_gt)
    precision = np.count_nonzero(accuracy < miu) / num_samples
    accuracy_l2 = accuracy ** 2
    accuracy_l1 = accuracy_l1.mean()
    accuracy_l2 = accuracy_l2.mean()
    normals_accuracy = normals_accuracy.mean()
    distance1 = (completeness_l1 + accuracy_l1) / 2
    distance2 = (completeness_l2 + accuracy_l2) / 2
    normals_correctness = (normals_completeness + normals_accuracy) / 2
    if precision < 1e-7 and recall < 1e-7:
        f_score_pts = 0
    else:
        f_score_pts = 2 * precision * recall / (precision + recall)
    return distance1, distance2, normals_correctness, f_score_pts


# calculate f-score based on sdf value
def calculate_f_score(mesh_gt, mesh_pred, miu=5e-3, num_samples=10 ** 5):
    points_pred = mesh_pred.sample(num_samples).astype(np.float32)
    sdf_pred = pysdf.SDF(mesh_pred.vertices, mesh_pred.faces)
    points_gt = mesh_gt.sample(num_samples).astype(np.float32)
    sdf_gt = pysdf.SDF(mesh_gt.vertices, mesh_gt.faces)
    precision_sdf = np.abs(sdf_gt(points_pred)).astype(np.float32)
    precision = np.count_nonzero(precision_sdf < miu) / num_samples
    recall_sdf = np.abs(sdf_pred(points_gt)).astype(np.float32)
    recall = np.count_nonzero(recall_sdf < miu) / num_samples
    if precision < 1e-7 and recall < 1e-7:
        score = 0
        return score
    score = 2 * precision * recall / (precision + recall)
    return score
