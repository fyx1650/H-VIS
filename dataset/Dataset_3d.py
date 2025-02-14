import numpy as np
import os
import torch
from torch.utils.data import Dataset
# from pytorch3d.ops.knn import knn_points, knn_gather
from scipy.spatial import cKDTree
from tqdm import tqdm

import trimesh
import open3d as o3d

import pysdf
import random


def map_color(value, cmap_name='viridis', vmin=None, vmax=None):
    # value: [N], float
    # return: RGB, [N, 3], float in [0, 1]
    import matplotlib.cm as cm
    if vmin is None: vmin = value.min()
    if vmax is None: vmax = value.max()
    if vmax == vmin:
        value += 0.2
    else:
        value = (value - vmin) / (vmax - vmin)  # range in [0, 1]
    cmap = cm.get_cmap(cmap_name)
    rgb = cmap(value)[:, :3]  # will return rgba, we take only first 3 so we get rgb
    return rgb


def plot_pointcloud(pc, sdfs):
    # pc: [N, 3]
    # sdfs: [N, 1]
    color = map_color(sdfs.squeeze(1))
    pc = trimesh.PointCloud(pc, color)
    trimesh.Scene([pc]).show()


# ----simple geometric downsampling based on discrete curvature---- #
def geometry_sample(points, normals, num_samples=10000, num_samples_down=50000):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    pcd.normals = o3d.utility.Vector3dVector(np.asarray(normals))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(pcd.points))
    for i in range(len(pcd.points)):
        [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], 30)
        neighbors = np.asarray(pcd.points)[idx, :]
        covariance_matrix = np.cov(neighbors.T)
        eigenvalues, _ = np.linalg.eigh(covariance_matrix)
        curvatures[i] = abs(eigenvalues[0] / np.sum(eigenvalues))
    sampling_probabilities = curvatures / curvatures.sum()
    indices = np.random.choice(num_samples, size=num_samples_down, replace=False, p=sampling_probabilities)
    points_sample = points[indices, :]
    normals_sample = normals[indices, :]
    return points_sample, normals_sample


class SDFDataset_3D(Dataset):
    def __init__(self, device, pts_path, mesh_path=None, size=500, num_samples=100000, batchsize=5000, online=True,
                 model_scale=1, space_scale=1):
        super().__init__()
        self.device = device
        self.pts_path = pts_path
        self.mesh_path = mesh_path
        self.batchsize = batchsize
        self.size = size
        self.online = online
        self.model_scale = model_scale
        self.space_scale = space_scale

        # load points cloud
        self.pc = trimesh.load(self.pts_path, force='mesh')
        if self.mesh_path is not None:
            self.mesh_gt = trimesh.load(self.mesh_path, force='mesh')
        else:
            self.mesh_gt = None

        # ---- normalize to [-1, 1] ---- #
        vs = self.pc.vertices
        v_center = self.pc.centroid
        vs = vs - v_center
        v_scale = np.abs(np.asarray(vs)).max()
        vs = vs / v_scale * self.model_scale
        self.pc.vertices = vs
        self.center = v_center
        self.scale = v_scale
        self.vertices = np.asarray(vs)
        self.exist_normals = True

        print(f"[INFO] PointCloud: {self.pc.vertices.shape}")

        self.num_samples = self.pc.vertices.shape[0]
        self.points_mnfld = np.asarray(self.pc.vertices)
        self.normals_mnfld = np.zeros_like(self.points_mnfld)

        mnfld_gt_n = self.read_normals(pts_path)
        if mnfld_gt_n.shape[0] > 0:
            self.normals_mnfld = mnfld_gt_n
        else:
            self.exist_normals = False

        # ----knn(pytorch3d ver.)---- #
        # pts = torch.from_numpy(self.points_mnfld).unsqueeze(0).to(self.device)
        # pnn = knn_points(pts, pts, K=51)
        # self.sigmas = np.sqrt(pnn.dists[..., -1].squeeze().detach().cpu().numpy())

        # ----knn(scipy ver.)---- #
        self.kdtree = cKDTree(self.points_mnfld)
        dist, _ = self.kdtree.query(self.points_mnfld, k=51)
        self.sigmas = dist[:, -1]

        bound_min = np.min(self.points_mnfld, axis=0)
        bound_max = np.max(self.points_mnfld, axis=0)
        self.bbox_model = np.stack(
            (bound_min - 0.05 * (bound_max - bound_min), bound_max + 0.05 * (bound_max - bound_min))).astype(np.float32)
        self.bbox = np.array([[-space_scale, -space_scale, -space_scale], [space_scale, space_scale, space_scale]]).astype(np.float32)

    def __len__(self):
        return self.size

    def read_normals(self, path):
        mesh = o3d.io.read_point_cloud(path)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        return normals

    def __getitem__(self, index):
        choose_array = np.random.choice(self.num_samples, self.batchsize, replace=False)
        choose_array2 = np.random.choice(self.batchsize, self.batchsize * 4 // 4, replace=False)
        sigma_mnfld = self.sigmas[choose_array][choose_array2][:, None]
        points_mnfld = self.points_mnfld[choose_array, :]
        normals_mnfld = self.normals_mnfld[choose_array, :]
        # pts = torch.from_numpy(self.points_mnfld).unsqueeze(0).to(self.device)
        guassian_vecs = np.random.randn(sigma_mnfld.shape[0], 3)

        # ----clip the points near inside the bounding box---- #
        points_mnfld_temp = points_mnfld[choose_array2, :]
        points_near_temp = points_mnfld_temp + sigma_mnfld * guassian_vecs

        # # 1.clip by the boundary of the bounding box
        # points_near = np.where(points_near_temp < self.bbox[0][None, :],
        #                        np.tile(self.bbox[0] + 0.005 * (self.bbox[1] - self.bbox[0])[None, :],
        #                                (points_near_temp.shape[0], 1)),
        #                        points_near_temp)
        # points_near = np.where(points_near_temp > self.bbox[1][None, :],
        #                        np.tile(self.bbox[1] - 0.005 * (self.bbox[1] - self.bbox[0])[None, :],
        #                                (points_near_temp.shape[0], 1)),
        #                        points_near_temp)

        # 2.clip along the line from points mnfld to points near
        center = (self.bbox[0] + self.bbox[1]) / 2
        half_range = (self.bbox[1] - self.bbox[0]) / 2

        points_near_temp1 = np.where(np.abs(points_near_temp - center) > half_range,
                                     np.abs(center + np.sign(points_near_temp - center) * half_range -
                                            points_mnfld_temp), 1)
        points_near_temp2 = np.where(np.abs(points_near_temp - center) > half_range,
                                     np.abs(points_near_temp - points_mnfld_temp), 1)
        points_near_sc = np.max(points_near_temp2 / (points_near_temp1 + 1e-6), axis=-1, keepdims=True) * 1.05
        points_near = (points_mnfld_temp + sigma_mnfld / points_near_sc *
                       guassian_vecs)

        # ----test whether points near inside the bounding box---- #
        if (np.count_nonzero(np.where(points_near > self.bbox[1])) > 0 or
            np.count_nonzero(np.where(points_near < self.bbox[0])) > 0):
            print(np.where(points_near > self.bbox[1]), np.where(points_near < self.bbox[0]))
            raise ValueError('unexpected points')

        # ----knn(pytorch3d ver.)---- #
        # pts1 = torch.from_numpy(points_near).unsqueeze(0).to(self.device)
        # pnn1 = knn_points(pts1, pts, K=1)
        # sigmas1 = np.sqrt(pnn1.dists[..., -1].squeeze().detach().cpu().numpy())[:, None]
        # points_target_near = knn_gather(pts, pnn1.idx).squeeze().detach().cpu().numpy()  # closest points to points near in points mnfld

        # ----knn(scipy ver.)---- #
        # dist1, idx1 = self.kdtree.query(points_near, k=1)
        # sigmas1 = dist1[:, None]
        # points_target_near = self.points_mnfld[idx1, :]

        # ----knn(dummy ver.)---- #
        sigmas1 = np.zeros((points_near.shape[0], 1), dtype=np.float32)  # dummy, save time
        points_target_near = np.zeros_like(points_near)  # dummy, save time

        points_nonmnfld = self.bbox[0] + (self.bbox[1] - self.bbox[0]) * np.random.rand(points_mnfld.shape[0], 3)

        # ----knn(pytorch3d ver.)---- #
        # pts2 = torch.from_numpy(points_nonmnfld).unsqueeze(0).to(self.device)
        # pnn2 = knn_points(pts2, pts, K=1)
        # points_target_nonmnfld = knn_gather(pts, pnn2.idx).squeeze().detach().cpu().numpy().astype(np.float32)

        # ----knn(scipy ver.)---- #
        # _, idx2 = self.kdtree.query(points_nonmnfld, k=1)
        # points_target_nonmnfld = self.points_mnfld[idx2, :]

        # ----knn(dummy ver.)---- #
        points_target_nonmnfld = np.zeros_like(points_nonmnfld)  # dummy, save time

        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        # sdfs_nonmnfld = -self.sdf_fn(points_nonmnfld)[:, None]
        # sdfs_near = -self.sdf_fn(points_near)[:, None]
        sdfs_nonmnfld = np.zeros((points_nonmnfld.shape[0], 1))  # fake
        sdfs_near = np.zeros((points_near.shape[0], 1))  # fake
        normals_nonmnfld = np.zeros_like(points_nonmnfld)  # fake,hard to get at first
        normals_near = np.zeros_like(points_near)  # fake,hard to get at first

        points_mnfld = np.ascontiguousarray(points_mnfld).astype(np.float32)
        points_nonmnfld = np.ascontiguousarray(points_nonmnfld).astype(np.float32)
        points_near = np.ascontiguousarray(points_near).astype(np.float32)

        sdfs_mnfld = np.ascontiguousarray(sdfs_mnfld).astype(np.float32)
        sdfs_nonmnfld = np.ascontiguousarray(sdfs_nonmnfld).astype(np.float32)
        sdfs_near = np.ascontiguousarray(sdfs_near).astype(np.float32)
        sigmas1 = np.ascontiguousarray(sigmas1).astype(np.float32)
        points_target_near = np.ascontiguousarray(points_target_near).astype(np.float32)
        points_target_nonmnfld = np.ascontiguousarray(points_target_nonmnfld).astype(np.float32)

        normals_mnfld = np.ascontiguousarray(normals_mnfld).astype(np.float32)
        normals_nonmnfld = np.ascontiguousarray(normals_nonmnfld).astype(np.float32)
        normals_near = np.ascontiguousarray(normals_near).astype(np.float32)

        results = {
            'points_mnfld': points_mnfld,
            'sdfs_mnfld': sdfs_mnfld,
            'normals_mnfld': normals_mnfld,
            'points_nonmnfld': points_nonmnfld,
            'sdfs_nonmnfld': sdfs_nonmnfld,
            'normals_nonmnfld': normals_nonmnfld,
            'points_near': points_near,
            'sdfs_near': sdfs_near,
            'normals_near': normals_near,
            'vertices': self.vertices,
            'exist_normals': self.exist_normals,
            'sigmas1': sigmas1,
            'points_target_near': points_target_near,
            'points_target_nonmnfld': points_target_nonmnfld
        }

        # plot_pointcloud(points_near, sdfs_near)
        return results


# This dataset is just for sdf regression task
class SDFDataset_3D_online(Dataset):
    def __init__(self, device, pts_path, mesh_path, size=500, num_samples=100000, batchsize=5000, online=True,
                 model_scale=1, space_scale=1):
        super().__init__()
        self.device = device
        self.pts_path = pts_path
        self.mesh_path = mesh_path
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.size = size
        self.online = online
        self.model_scale = model_scale
        self.space_scale = space_scale

        # load npz
        if self.pts_path[-3:] == 'npz':
            files_all = np.load(self.pts_path)
            self.points_mnfld = files_all['mnfld']
            self.points_near = files_all['near'][:, :-1]
            self.sdf_near = files_all['near'][:, -1][:, None]
            self.points_far = files_all['far'][:, :-1]
            self.sdf_far = files_all['far'][:, -1][:, None]

        self.normals_mnfld = np.zeros_like(self.points_mnfld)

        # load mesh
        self.mesh_gt = trimesh.load(self.mesh_path, force='mesh')
        self.mesh_sample = self.mesh_gt.copy()

        # ---- normalize to [-1, 1] ---- #
        vs = self.mesh_gt.vertices
        v_center = self.mesh_gt.centroid
        vs = vs - v_center
        v_scale = np.abs(np.asarray(vs)).max()
        vs = vs / v_scale * self.model_scale
        self.mesh_sample.vertices = vs
        self.center = v_center
        self.scale = v_scale
        self.vertices = np.asarray(vs)
        self.exist_normals = True

        if self.pts_path[-3:] != 'npz':
            """
            Note that the pysdf function can't be serialized, and it could cost much time if the mesh is too large, 
            so we suggest to store the data in a npz file in advance, and we provide a function 'generate_npz' to handle
            it in utils.py.
            """
            self.sdf_fn = pysdf.SDF(vs, self.mesh_gt.faces)

        print(f"[INFO] Mesh: {self.mesh_gt.vertices.shape}")

        self.exist_normals = False

        bound_min = np.min(self.vertices, axis=0)
        bound_max = np.max(self.vertices, axis=0)
        self.bbox_model = np.stack(
            (bound_min - 0.05 * (bound_max - bound_min), bound_max + 0.05 * (bound_max - bound_min))).astype(np.float32)
        self.bbox = np.array([[-space_scale, -space_scale, -space_scale], [space_scale, space_scale, space_scale]]).astype(np.float32)

    def __len__(self):
        return self.size

    def read_normals(self, path):
        mesh = o3d.io.read_point_cloud(path)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        return normals

    def __getitem__(self, index):
        if self.pts_path[-3:] == 'npz':
            choose_array = np.random.choice(np.arange(self.points_mnfld.shape[0]), self.num_samples,
                                                 replace=False)
            points_mnfld = self.points_mnfld[choose_array[: self.num_samples // 2], :]
            normals_mnfld = np.zeros_like(points_mnfld)  # fake
            points_near = self.points_near[choose_array[self.num_samples // 2: self.num_samples * 7 // 8], :]
            sdfs_near = self.sdf_near[choose_array[self.num_samples // 2: self.num_samples * 7 // 8], :]
            points_nonmnfld = self.points_far[choose_array[self.num_samples * 7 // 8: self.num_samples], :]
            sdfs_nonmnfld = self.sdf_far[choose_array[self.num_samples * 7 // 8: self.num_samples], :]
        else:
            p_m, faces = self.mesh_sample.sample(self.num_samples, return_index=True)
            points_mnfld = np.asarray(p_m)
            normals_mnfld = np.asarray(self.mesh_sample.face_normals[faces])
            p_m_near, _ = self.mesh_sample.sample(self.num_samples * 3 // 4, return_index=True)
            points_near = np.asarray(p_m_near) + 0.01 * np.random.randn(self.num_samples * 3 // 4, 3)
            points_nonmnfld = self.bbox[0] + (self.bbox[1] - self.bbox[0]) * np.random.rand(self.num_samples // 4, 3)
            sdfs_nonmnfld = self.sdf_fn(points_nonmnfld)[:, None]
            sdfs_near = self.sdf_fn(points_near)[:, None]

        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        sigmas1 = np.zeros((self.num_samples * 3 // 4, 1))
        points_target_near = np.zeros_like(points_near)  # dummy, save time
        points_target_nonmnfld = np.zeros_like(points_nonmnfld)  # dummy, save time
        normals_nonmnfld = np.zeros_like(points_nonmnfld)  # fake,hard to get at first
        normals_near = np.zeros_like(points_near)  # fake,hard to get at first

        if (np.count_nonzero(np.where(points_near > self.bbox[1])) > 0 or
            np.count_nonzero(np.where(points_near < self.bbox[0])) > 0):
            print(np.where(points_near > self.bbox[1]), np.where(points_near < self.bbox[0]))
            raise ValueError('unexpected points')

        points_mnfld = np.ascontiguousarray(points_mnfld).astype(np.float32)
        points_nonmnfld = np.ascontiguousarray(points_nonmnfld).astype(np.float32)
        points_near = np.ascontiguousarray(points_near).astype(np.float32)

        sdfs_mnfld = np.ascontiguousarray(sdfs_mnfld).astype(np.float32)
        sdfs_nonmnfld = np.ascontiguousarray(sdfs_nonmnfld).astype(np.float32)
        sdfs_near = np.ascontiguousarray(sdfs_near).astype(np.float32)
        sigmas1 = np.ascontiguousarray(sigmas1).astype(np.float32)
        points_target_near = np.ascontiguousarray(points_target_near).astype(np.float32)
        points_target_nonmnfld = np.ascontiguousarray(points_target_nonmnfld).astype(np.float32)

        normals_mnfld = np.ascontiguousarray(normals_mnfld).astype(np.float32)
        normals_nonmnfld = np.ascontiguousarray(normals_nonmnfld).astype(np.float32)
        normals_near = np.ascontiguousarray(normals_near).astype(np.float32)

        results = {
            'points_mnfld': points_mnfld,
            'sdfs_mnfld': sdfs_mnfld,
            'normals_mnfld': normals_mnfld,
            'points_nonmnfld': points_nonmnfld,
            'sdfs_nonmnfld': sdfs_nonmnfld,
            'normals_nonmnfld': normals_nonmnfld,
            'points_near': points_near,
            'sdfs_near': sdfs_near,
            'normals_near': normals_near,
            'vertices': self.vertices,
            'exist_normals': self.exist_normals,
            'sigmas1': sigmas1,
            'points_target_near': points_target_near,
            'points_target_nonmnfld': points_target_nonmnfld
        }

        # plot_pointcloud(points_near, sdfs_near)
        return results
