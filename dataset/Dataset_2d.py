import numpy as np
import time

import torch
from torch.utils.data import Dataset
from abc import abstractmethod
from matplotlib import pyplot as plt
from matplotlib.path import Path
# from pytorch3d.ops.knn import knn_points, knn_gather
from tqdm import tqdm
from scipy.spatial import cKDTree
import plotly.io
from plotly import graph_objects as go


class SDFDataset_2D(Dataset):
    def __init__(self, device, size=100, num_samples=2 ** 18, batchsize=2 ** 18, online=False, space_scale=1):
        super().__init__()
        self.device = device
        self.size = size
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.online = online
        self.space_scale = space_scale
        self.vertices = self.get_new_vertices()
        self.lines = self.get_line_props()
        self.exist_normals = True
        self.points_mnfld, self.normals_mnfld, self.sdfs_mnfld = self.get_points_mnfld_with_normal_and_sdfs()
        self.points_mnfld = np.where(np.abs(self.points_mnfld) < 1e-7, 0, self.points_mnfld)
        # pts = torch.from_numpy(self.points_mnfld).unsqueeze(0).to(self.device)
        # pnn = knn_points(pts, pts, K=51)
        # self.sigmas = np.sqrt(pnn.dists[..., -1].squeeze().detach().cpu().numpy())
        self.kdtree = cKDTree(self.points_mnfld)
        dist, _ = self.kdtree.query(self.points_mnfld, k=51)
        self.sigmas = dist[:, -1]
        bound_min = np.min(self.points_mnfld, axis=0)
        bound_max = np.max(self.points_mnfld, axis=0)
        self.bbox_model = np.array([bound_min - 0.05 * (bound_max - bound_min), bound_max + 0.05 * (bound_max - bound_min)]).astype(np.float32)
        self.bbox = np.array([[-space_scale, -space_scale], [space_scale, space_scale]]).astype(np.float32)

    @abstractmethod
    def get_new_vertices(self, *args):
        pass

    @abstractmethod
    def get_line_props(self, *args):
        pass

    @abstractmethod
    def get_points_mnfld_with_normal_and_sdfs(self):
        pass

    def get_points_nonmnfld(self, points_mnfld):
        points_nonmnfld = self.bbox[0] + (self.bbox[1] - self.bbox[0]) * np.random.rand(points_mnfld.shape[0], 2)
        return points_nonmnfld

    @abstractmethod
    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        pass

    def plot_sdf_gt(self, res=512, save_path=None):
        x = np.linspace(self.bbox_model[0, 0], self.bbox_model[1, 0], res + 1)
        y = np.linspace(self.bbox_model[0, 1], self.bbox_model[1, 1], res + 1)
        xx, yy = np.meshgrid(x, y, indexing='xy')
        points_all = np.concatenate((xx[..., None], yy[..., None]), axis=-1)
        points_all = np.reshape(points_all, ((res + 1) ** 2, 2))
        _, sdfs = self.get_points_nonmnfld_with_normal_and_sdfs(points_all)
        z = -np.reshape(np.transpose(sdfs), (res+1, res+1))
        layout = go.Layout(width=2160, height=2160,
                           xaxis=dict(side="bottom", range=[self.bbox_model[0, 0], self.bbox_model[1, 0]], showticklabels=False),
                           yaxis=dict(side="left", range=[self.bbox_model[0, 1], self.bbox_model[1, 1]], showticklabels=False),
                           scene=dict(xaxis=dict(range=[self.bbox_model[0, 0], self.bbox_model[1, 0]], autorange=False),
                                      yaxis=dict(range=[self.bbox_model[0, 1], self.bbox_model[1, 1]], autorange=False),
                                      aspectratio=dict(x=1, y=1)),
                           showlegend=False,
                           # title=dict(text='distance map', y=0.95, x=0.5, xanchor='center', yanchor='middle',
                           #            font=dict(family='Serif', size=24, color='red'))
                           )
        traces = []
        # start_sdf = np.min(z)
        # end_sdf = np.max(z)
        traces.append(go.Contour(x=x, y=y, z=z,
                                 colorscale='Geyser',
                                 colorbar=dict(tickfont=dict(size=75), thickness=75, tickwidth=3),
                                 # autocontour=True,
                                 contours=dict(
                                     start=-0.3,
                                     end=0.3,
                                     size=0.025,
                                 ),
                                 line=dict(width=3),
                                 showscale=True
                                 ))  # contour trace
        vert = np.concatenate((self.vertices, self.vertices[0][None, :]))
        traces.append(go.Scatter(x=vert[:, 0], y=vert[:, 1], mode='lines',
                      line=dict(width=10, color='black')))
        fig = go.Figure(data=traces, layout=layout)
        plotly.io.write_image(fig, save_path)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.online:
            from pytorch3d.ops.knn import knn_points, knn_gather
            points_mnfld, normals_mnfld, sdfs_mnfld = self.get_points_mnfld_with_normal_and_sdfs()
            points_mnfld = np.where(np.abs(points_mnfld) < 1e-7, 0, points_mnfld)

            pts = torch.from_numpy(points_mnfld).unsqueeze(0).to(self.device)
            pnn = knn_points(pts, pts, K=51)
            sigmas = np.sqrt(pnn.dists[..., -1].squeeze().detach().cpu().numpy())
            choose_array2 = np.random.choice(self.batchsize, self.batchsize * 4 // 4, replace=False)
            sigma_mnfld = sigmas[choose_array2, None]
            guassian_vecs = np.random.randn(sigma_mnfld.shape[0], 2)
        else:
            choose_array = np.random.choice(self.points_mnfld.shape[0], self.batchsize, replace=False)
            points_mnfld = self.points_mnfld[choose_array, :]
            sdfs_mnfld = self.sdfs_mnfld[choose_array, :]
            normals_mnfld = self.normals_mnfld[choose_array, :]
            # pts = torch.from_numpy(self.points_mnfld).unsqueeze(0).to(self.device)

            choose_array2 = np.random.choice(self.batchsize, self.batchsize * 4 // 4, replace=False)
            sigma_mnfld = self.sigmas[choose_array][choose_array2, None]
            guassian_vecs = np.random.randn(sigma_mnfld.shape[0], 2)

        # ----clip the points_near with the bounding box---- #
        points_mnfld_temp = points_mnfld[choose_array2, :]
        points_near_temp = points_mnfld_temp + sigma_mnfld * guassian_vecs

        # 1.clip by the boundary of the bounding box
        # points_near = np.where(points_near_temp < self.bbox[0][None, :],
        #                        np.tile(self.bbox[0] + 0.005 * (self.bbox[1] - self.bbox[0])[None, :],
        #                                (points_near_temp.shape[0], 1)),
        #                        points_near_temp)
        # points_near = np.where(points_near_temp > self.bbox[1][None, :],
        #                        np.tile(self.bbox[1] - 0.005 * (self.bbox[1] - self.bbox[0])[None, :],
        #                                (points_near_temp.shape[0], 1)),
        #                        points_near_temp)

        # 2.clip along the line from points_mnfld to points_near
        center = (self.bbox[0] + self.bbox[1]) / 2
        half_range = (self.bbox[1] - self.bbox[0]) / 2

        points_near_temp1 = np.where(np.abs(points_near_temp - center) > half_range,
                                     np.abs(center + np.sign(points_near_temp - center) * half_range -
                                            points_mnfld_temp), 1)
        points_near_temp2 = np.where(np.abs(points_near_temp - center) > half_range,
                                     np.abs(points_near_temp - points_mnfld_temp), 1)
        points_near_sc = np.max(points_near_temp2 / (points_near_temp1 + 1e-5), axis=-1, keepdims=True) * 1.01
        points_near = (points_mnfld_temp + sigma_mnfld / points_near_sc *
                       guassian_vecs)
        normals_near, sdfs_near = self.get_points_nonmnfld_with_normal_and_sdfs(points_near)

        # ----test whether points near inside the bounding box---- #
        if (np.count_nonzero(np.where(points_near > self.bbox[1])) > 0 or
                np.count_nonzero(np.where(points_near < self.bbox[0])) > 0):
            print(np.where(points_near > self.bbox[1]), np.where(points_near < self.bbox[0]))
            raise ValueError('unexpected points')

        # pts1 = torch.from_numpy(points_near).unsqueeze(0).to(self.device)
        # pnn1 = knn_points(pts1, pts, K=1)
        # sigmas1 = np.sqrt(pnn1.dists[..., -1].squeeze().detach().cpu().numpy())[:, None]
        # points_target_near = knn_gather(pts, pnn1.idx).squeeze().detach().cpu().numpy()  # closest points to points near in points mnfld
        dist1, idx1 = self.kdtree.query(points_near, k=1)
        sigmas1 = dist1[:, None]
        points_target_near = self.points_mnfld[idx1, :]
        # sigmas1 = np.zeros((points_near.shape[0], 1), dtype=np.float32)  # dummy, save time
        # points_target_near = np.zeros_like(points_near)  # dummy, save time

        points_nonmnfld = self.get_points_nonmnfld(points_mnfld)
        normals_nonmnfld, sdfs_nonmnfld = self.get_points_nonmnfld_with_normal_and_sdfs(points_nonmnfld)
        # pts2 = torch.from_numpy(points_nonmnfld).unsqueeze(0).to(self.device)
        # pnn2 = knn_points(pts2, pts, K=1)
        # points_target_nonmnfld = knn_gather(pts, pnn2.idx).squeeze().detach().cpu().numpy()  # closest points to points nonmnfld in points mnfld
        _, idx2 = self.kdtree.query(points_nonmnfld, k=1)
        points_target_nonmnfld = self.points_mnfld[idx2, :]
        # points_target_nonmnfld = np.zeros_like(points_nonmnfld)  # dummy, save time

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

        return results


class Circle(SDFDataset_2D):

    def __init__(self, *args, radius=0.8, cx=0.0, cy=0.0):
        self.radius = radius
        self.center = np.asarray([[cx, cy]])
        super().__init__(*args)

    def get_new_vertices(self):
        arr = np.linspace(0, 2 * np.pi, num=128, endpoint=False)[:, None]
        vertices = self.center + np.concatenate((self.radius * np.cos(arr), self.radius * np.sin(arr)), axis=1)
        return vertices

    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices) - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines

    def get_points_mnfld_with_normal_and_sdfs(self):
        theta = 2 * np.pi * np.random.rand(self.num_samples, 1)
        points_mnfld = self.center + np.concatenate([self.radius * np.cos(theta), self.radius * np.sin(theta)], axis=1)
        normals_mnfld = (points_mnfld - self.center) / np.linalg.norm((points_mnfld - self.center), axis=1, keepdims=True)
        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        return points_mnfld, normals_mnfld, sdfs_mnfld

    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        sdfs_nonmnfld = np.linalg.norm(points_nonmnfld, axis=1, keepdims=True) - self.radius
        normals_nonmnfld = (points_nonmnfld - self.center) / np.linalg.norm((points_nonmnfld - self.center), axis=1, keepdims=True)
        return normals_nonmnfld, sdfs_nonmnfld


class double_Circle(SDFDataset_2D):

    def __init__(self, *args, radius=0.2, cx1=-0.25, cy1=0.0, cx2=0.25, cy2=0.0):
        self.radius = radius
        self.center1 = np.asarray([[cx1, cy1]])
        self.center2 = np.asarray([[cx2, cy2]])
        super().__init__(*args)

    def get_new_vertices(self):
        arr = np.linspace(0, 2 * np.pi, num=128, endpoint=False)[:, None]
        vertices1 = self.center1 + np.concatenate((self.radius * np.cos(arr), self.radius * np.sin(arr)), axis=1)
        vertices2 = self.center2 + np.concatenate((self.radius * np.cos(arr), self.radius * np.sin(arr)), axis=1)
        vertices = np.concatenate((vertices1, vertices2), axis=0)
        return vertices

    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        num_v = len(self.vertices)
        for start_idx, start_point in enumerate(self.vertices[:num_v // 2, :]):
            if start_idx < num_v // 2:
                end_idx = 0 if start_idx == num_v // 2 - 1 else start_idx + 1
            else:
                end_idx = num_v // 2 if start_idx == num_v - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])
        return lines

    def get_points_mnfld_with_normal_and_sdfs(self):
        theta = 2 * np.pi * np.random.rand(self.num_samples // 2, 1)
        points_mnfld1 = self.center1 + np.concatenate([self.radius * np.cos(theta), self.radius * np.sin(theta)], axis=1)
        points_mnfld2 = self.center2 + np.concatenate([self.radius * np.cos(theta), self.radius * np.sin(theta)],
                                                      axis=1)
        points_mnfld = np.concatenate((points_mnfld1, points_mnfld2), axis=0)
        normals_mnfld1 = (points_mnfld1 - self.center1) / np.linalg.norm((points_mnfld1 - self.center1), axis=1, keepdims=True)
        normals_mnfld2 = (points_mnfld2 - self.center2) / np.linalg.norm((points_mnfld2 - self.center2), axis=1,
                                                                         keepdims=True)
        normals_mnfld = np.concatenate((normals_mnfld1, normals_mnfld2), axis=0)
        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        return points_mnfld, normals_mnfld, sdfs_mnfld

    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        sdfs_nonmnfld1 = np.linalg.norm(points_nonmnfld - self.center1, axis=1, keepdims=True) - self.radius
        sdfs_nonmnfld2 = np.linalg.norm(points_nonmnfld - self.center2, axis=1, keepdims=True) - self.radius
        sdfs_nonmnfld = np.where(sdfs_nonmnfld1 < sdfs_nonmnfld2, sdfs_nonmnfld1, sdfs_nonmnfld2)
        normals_nonmnfld1 = (points_nonmnfld - self.center1) / np.linalg.norm((points_nonmnfld - self.center1), axis=1, keepdims=True)
        normals_nonmnfld2 = (points_nonmnfld - self.center2) / np.linalg.norm((points_nonmnfld - self.center2), axis=1,
                                                                              keepdims=True)
        normals_nonmnfld = np.where(sdfs_nonmnfld1 < sdfs_nonmnfld2, normals_nonmnfld1, normals_nonmnfld2)
        return normals_nonmnfld, sdfs_nonmnfld


# reference:StEik:https://github.com/sunyx523/StEik/tree/main
class Polygon(SDFDataset_2D):

    def __init__(self, *args, vertices, line_sample_type='uniform'):
        self.line_sample_type = line_sample_type
        self.vertices = np.array(vertices)
        super().__init__(*args)

    def get_new_vertices(self, *args):
        return self.vertices

    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices) - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines

    def get_points_mnfld_with_normal_and_sdfs(self):
        num_samples_mnfld = self.num_samples - len(self.vertices)
        if num_samples_mnfld < 0:
            raise Warning("Fewer points to sample than polygon vertices. Please change the number of points")

        sample_prob = self.lines['line_length'] / np.sum(self.lines['line_length'])
        points_per_segment = np.floor(num_samples_mnfld * sample_prob).astype(np.int32)
        points_leftover = int(num_samples_mnfld - points_per_segment.sum())
        if not points_leftover == 0:
            for j in range(points_leftover):
                actual_prob = points_per_segment / points_per_segment.sum()
                prob_diff = sample_prob - actual_prob
                add_idx = np.argmax(prob_diff)
                points_per_segment[add_idx] = points_per_segment[add_idx] + 1

        points_mnfld = []
        normals_mnfld = []
        for point_idx, point in enumerate(self.vertices):
            l1_idx = len(self.vertices) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines['nl'][l1_idx] + self.lines['nl'][l2_idx]
            points_mnfld.append(point)
            normals_mnfld.append(n / np.linalg.norm(n))
        points_mnfld = np.array(points_mnfld)
        normals_mnfld = np.array(normals_mnfld)

        for line_idx in range(len(self.lines['A'])):
            if self.line_sample_type == 'uniform':
                t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
            else:
                t = np.random.uniform(0, 1, points_per_segment[line_idx])
            p1 = np.array(self.vertices[self.lines['start_idx'][line_idx]])
            p2 = np.array(self.vertices[self.lines['end_idx'][line_idx]])
            points_mnfld = np.concatenate([points_mnfld, p1 + t[:, None] * (p2 - p1)], axis=0)
            normals_mnfld = np.concatenate([normals_mnfld, np.tile(self.lines['nl'][line_idx],
                                                                   [points_per_segment[line_idx], 1])], axis=0)
        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        return points_mnfld, normals_mnfld, sdfs_mnfld

    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        # iterate over all the lines and  finds the minimum distance between all points and line segments
        # good explanation ref : https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
        p1 = self.vertices[self.lines['start_idx'], :]
        p2 = self.vertices[self.lines['end_idx'], :]
        p1p2 = np.array(self.lines['direction'])
        p1p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p1
        p2p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p2

        r = (p1p2 * p1p).sum(-1) / np.array(self.lines['line_length'])
        d1 = np.linalg.norm(p1p, axis=-1)
        d2 = np.linalg.norm(p2p, axis=-1)
        dp = np.sqrt(np.abs(np.square(d1) - np.square(r * np.array(self.lines['line_length']))))

        d = np.where(r < 0, d1, np.where(r > 1, d2, dp))
        sdfs_nonmnfld = np.min(d, axis=-1, keepdims=True)
        idx = np.argmin(d, axis=-1)
        # compute normal vector
        polygon_path = Path(self.vertices)
        points_in_polygon = polygon_path.contains_points(points_nonmnfld)
        points_sign = np.where(points_in_polygon, -1, 1)
        lp = points_sign[:, None] * np.array(self.lines['nl'])[idx, :]
        rr = np.take_along_axis(r, idx[:, None], axis=1)
        normals_nonmnfld = np.where(rr < 0, np.take_along_axis(p1p, idx[:, None, None], axis=1).squeeze(),
                                    np.where(rr > 1, np.take_along_axis(p2p, idx[:, None, None], axis=1).squeeze(),
                                             lp))
        # normals_nonmnfld = np.take_along_axis(n, idx[:, None, None], axis=1).squeeze()
        normals_nonmnfld = points_sign[:, None] * normals_nonmnfld / np.linalg.norm(normals_nonmnfld, axis=-1,
                                                                                    keepdims=True)
        sdfs_nonmnfld = points_sign[:, None] * sdfs_nonmnfld

        return normals_nonmnfld, sdfs_nonmnfld


class Noisy_Circle(SDFDataset_2D):

    def __init__(self, *args, radius=0.8, cx=0.0, cy=0.0, line_sample_type='uniform'):
        self.radius = radius
        self.center = np.asarray([[cx, cy]])
        self.line_sample_type = line_sample_type
        super().__init__(*args)

    def get_new_vertices(self, *args):
        arr = np.linspace(0, 2 * np.pi, num=128, endpoint=False)[:, None]
        vertices_origin = self.center + np.concatenate((self.radius * np.cos(arr), self.radius * np.sin(arr)), axis=1)
        normals_origin = (vertices_origin - self.center) / np.linalg.norm((vertices_origin - self.center), axis=1, keepdims=True)

        # ----add_noise---- #
        vertices = vertices_origin + normals_origin * np.random.normal(0, 0.01, size=(vertices_origin.shape[0], 1))
        return vertices

    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices) - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines

    def get_points_mnfld_with_normal_and_sdfs(self):
        num_samples_mnfld = self.num_samples - len(self.vertices)
        if num_samples_mnfld < 0:
            raise Warning("Fewer points to sample than polygon vertices. Please change the number of points")
        sample_prob = self.lines['line_length'] / np.sum(self.lines['line_length'])
        points_per_segment = np.floor(num_samples_mnfld * sample_prob).astype(np.int32)
        points_leftover = int(num_samples_mnfld - points_per_segment.sum())
        if not points_leftover == 0:
            for j in range(points_leftover):
                actual_prob = points_per_segment / points_per_segment.sum()
                prob_diff = sample_prob - actual_prob
                add_idx = np.argmax(prob_diff)
                points_per_segment[add_idx] = points_per_segment[add_idx] + 1

        points_mnfld = []
        normals_mnfld = []
        for point_idx, point in enumerate(self.vertices):
            l1_idx = len(self.vertices) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines['nl'][l1_idx] + self.lines['nl'][l2_idx]
            points_mnfld.append(point)
            normals_mnfld.append(n / np.linalg.norm(n))
        points_mnfld = np.array(points_mnfld)
        normals_mnfld = np.array(normals_mnfld)

        for line_idx in range(len(self.lines['A'])):
            if self.line_sample_type == 'uniform':
                t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
            else:
                t = np.random.uniform(0, 1, points_per_segment[line_idx])
            p1 = np.array(self.vertices[self.lines['start_idx'][line_idx]])
            p2 = np.array(self.vertices[self.lines['end_idx'][line_idx]])
            points_mnfld = np.concatenate([points_mnfld, p1 + t[:, None] * (p2 - p1)], axis=0)
            normals_mnfld = np.concatenate([normals_mnfld, np.tile(self.lines['nl'][line_idx],
                                                                   [points_per_segment[line_idx], 1])], axis=0)
        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        return points_mnfld, normals_mnfld, sdfs_mnfld

    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        # iterate over all the lines and  finds the minimum distance between all points and line segments
        # good explenation ref : https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon
        p1 = self.vertices[self.lines['start_idx'], :]
        p2 = self.vertices[self.lines['end_idx'], :]
        p1p2 = np.array(self.lines['direction'])
        p1p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p1
        p2p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p2

        r = (p1p2 * p1p).sum(-1) / np.array(self.lines['line_length'])
        d1 = np.linalg.norm(p1p, axis=-1)
        d2 = np.linalg.norm(p2p, axis=-1)
        dp = np.sqrt(np.abs(np.square(d1) - np.square(r * np.array(self.lines['line_length']))))

        d = np.where(r < 0, d1, np.where(r > 1, d2, dp))
        sdfs_nonmnfld = np.min(d, axis=-1, keepdims=True)
        idx = np.argmin(d, axis=-1)
        # compute normal vector
        polygon_path = Path(self.vertices)
        points_in_polygon = polygon_path.contains_points(points_nonmnfld)
        points_sign = np.where(points_in_polygon, -1, 1)
        lp = points_sign[:, None] * np.array(self.lines['nl'])[idx, :]
        rr = np.take_along_axis(r, idx[:, None], axis=1)
        normals_nonmnfld = np.where(rr < 0, np.take_along_axis(p1p, idx[:, None, None], axis=1).squeeze(),
                                    np.where(rr > 1, np.take_along_axis(p2p, idx[:, None, None], axis=1).squeeze(),
                                             lp))
        normals_nonmnfld = points_sign[:, None] * normals_nonmnfld / np.linalg.norm(normals_nonmnfld, axis=-1, keepdims=True)
        sdfs_nonmnfld = points_sign[:, None] * sdfs_nonmnfld

        return normals_nonmnfld, sdfs_nonmnfld


class Noisy_Polygon(SDFDataset_2D):

    def __init__(self, *args, vertices, line_sample_type='uniform'):
        self.vertices_origin = np.array(vertices)
        self.lines_origin = self.get_line_props_new(self.vertices_origin)
        self.line_sample_type = line_sample_type
        super().__init__(*args)

    def get_line_props(self):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        for start_idx, start_point in enumerate(self.vertices):
            end_idx = 0 if start_idx == len(self.vertices) - 1 else start_idx + 1
            end_point = self.vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines

    def get_line_props_new(self, vertices):
        lines = {'A': [], 'B': [], 'C': [], 'nl': [], 'line_length': [], 'start_idx': [], 'end_idx': [],
                 'direction': []}
        for start_idx, start_point in enumerate(vertices):
            end_idx = 0 if start_idx == len(vertices) - 1 else start_idx + 1
            end_point = vertices[end_idx]
            # Compute standard form coefficients

            A = start_point[1] - end_point[1]
            B = end_point[0] - start_point[0]
            C = - (A * start_point[0] + B * start_point[1])
            line_length = np.sqrt(np.square(A) + np.square(B))
            direction = [(end_point[0] - start_point[0]) / line_length, (end_point[1] - start_point[1]) / line_length]
            nl = np.array([A / line_length, B / line_length])
            line_props = {'A': A, 'B': B, 'C': C, 'nl': nl, 'line_length': line_length,
                          'start_idx': start_idx, 'end_idx': end_idx, 'direction': direction}
            for key in lines.keys():
                lines[key].append(line_props[key])

        return lines

    def get_new_vertices(self, sample_num=20):
        points_per_segment = np.array([sample_num]*len(self.lines_origin['A']))

        points_mnfld = []
        normals_mnfld = []
        for point_idx, point in enumerate(self.vertices_origin):
            l1_idx = len(self.vertices_origin) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines_origin['nl'][l1_idx] + self.lines_origin['nl'][l2_idx]
            points_mnfld.append(point)
            normals_mnfld.append(n / np.linalg.norm(n))
        points_mnfld = np.array(points_mnfld)
        normals_mnfld = np.array(normals_mnfld)

        for line_idx in range(len(self.lines_origin['A'])):
            t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
            p1 = np.array(self.vertices_origin[self.lines_origin['start_idx'][line_idx]])
            p2 = np.array(self.vertices_origin[self.lines_origin['end_idx'][line_idx]])
            if line_idx == len(self.lines_origin['A']) - 1:
                points_mnfld = np.concatenate([points_mnfld, p1 + t[:, None] * (p2 - p1)], axis=0)
                normals_mnfld = np.concatenate([normals_mnfld, np.tile(self.lines_origin['nl'][line_idx],
                                                                       [points_per_segment[line_idx], 1])], axis=0)
            else:
                points_mnfld = np.concatenate([points_mnfld[:(line_idx * (sample_num + 1) + 1), :], p1 + t[:, None] *
                                               (p2 - p1), points_mnfld[(line_idx * (sample_num + 1) + 1):, :]], axis=0)
                normals_mnfld = np.concatenate([normals_mnfld[:(line_idx * (sample_num + 1) + 1), :],
                                                np.tile(self.lines_origin['nl'][line_idx], [points_per_segment[line_idx], 1]),
                                                normals_mnfld[(line_idx * (sample_num + 1) + 1):, :]], axis=0)
        edge_average = np.array(self.lines_origin['line_length']).mean() / sample_num
        vertices = points_mnfld + normals_mnfld * np.random.normal(0, 2 * edge_average, size=(points_mnfld.shape[0], 1))
        return vertices

    def get_points_mnfld_with_normal_and_sdfs(self):
        num_samples_mnfld = self.num_samples - len(self.vertices)
        if num_samples_mnfld < 0:
            raise Warning("Fewer points to sample than polygon vertices. Please change the number of points")

        sample_prob = self.lines['line_length'] / np.sum(self.lines['line_length'])
        points_per_segment = np.floor(num_samples_mnfld * sample_prob).astype(np.int32)
        points_leftover = int(num_samples_mnfld - points_per_segment.sum())
        if not points_leftover == 0:
            for j in range(points_leftover):
                actual_prob = points_per_segment / points_per_segment.sum()
                prob_diff = sample_prob - actual_prob
                add_idx = np.argmax(prob_diff)
                points_per_segment[add_idx] = points_per_segment[add_idx] + 1

        points_mnfld = []
        normals_mnfld = []
        for point_idx, point in enumerate(self.vertices):
            l1_idx = len(self.vertices) - 1 if point_idx == 0 else point_idx - 1
            l2_idx = point_idx
            n = self.lines['nl'][l1_idx] + self.lines['nl'][l2_idx]
            points_mnfld.append(point)
            normals_mnfld.append(n / np.linalg.norm(n))
        points_mnfld = np.array(points_mnfld)
        normals_mnfld = np.array(normals_mnfld)

        for line_idx in range(len(self.lines['A'])):
            if self.line_sample_type == 'uniform':
                t = np.linspace(0, 1, points_per_segment[line_idx] + 1, endpoint=False)[1:]
            else:
                t = np.random.uniform(0, 1, points_per_segment[line_idx])
            p1 = np.array(self.vertices[self.lines['start_idx'][line_idx]])
            p2 = np.array(self.vertices[self.lines['end_idx'][line_idx]])
            points_mnfld = np.concatenate([points_mnfld, p1 + t[:, None] * (p2 - p1)], axis=0)
            normals_mnfld = np.concatenate([normals_mnfld, np.tile(self.lines['nl'][line_idx],
                                                                   [points_per_segment[line_idx], 1])], axis=0)
        sdfs_mnfld = np.zeros((points_mnfld.shape[0], 1))
        return points_mnfld, normals_mnfld, sdfs_mnfld

    def get_points_nonmnfld_with_normal_and_sdfs(self, points_nonmnfld):
        # iterate over all the lines and  finds the minimum distance between all points and line segments
        # good explenation ref : https://stackoverflow.com/questions/10983872/distance-from-a-point-to-a-polygon

        p1 = self.vertices[self.lines['start_idx'], :]
        p2 = self.vertices[self.lines['end_idx'], :]
        p1p2 = np.array(self.lines['direction'])
        p1p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p1
        p2p = np.tile(points_nonmnfld[:, None, :], [1, len(self.lines['start_idx']), 1]) - p2

        r = (p1p2 * p1p).sum(-1) / np.array(self.lines['line_length'])
        d1 = np.linalg.norm(p1p, axis=-1)
        d2 = np.linalg.norm(p2p, axis=-1)
        dp = np.sqrt(np.abs(np.square(d1) - np.square(r * np.array(self.lines['line_length']))))

        d = np.where(r < 0, d1, np.where(r > 1, d2, dp))
        sdfs_nonmnfld = np.min(d, axis=-1, keepdims=True)
        idx = np.argmin(d, axis=-1)
        # compute normal vector
        polygon_path = Path(self.vertices)
        points_in_polygon = polygon_path.contains_points(points_nonmnfld)
        points_sign = np.where(points_in_polygon, -1, 1)
        lp = points_sign[:, None] * np.array(self.lines['nl'])[idx, :]
        rr = np.take_along_axis(r, idx[:, None], axis=1)
        normals_nonmnfld = np.where(rr < 0, np.take_along_axis(p1p, idx[:, None, None], axis=1).squeeze(),
                                    np.where(rr > 1, np.take_along_axis(p2p, idx[:, None, None], axis=1).squeeze(),
                                             lp))
        # normals_nonmnfld = np.take_along_axis(n, idx[:, None, None], axis=1).squeeze()
        normals_nonmnfld = points_sign[:, None] * normals_nonmnfld / np.linalg.norm(normals_nonmnfld, axis=-1,
                                                                                    keepdims=True)
        sdfs_nonmnfld = points_sign[:, None] * sdfs_nonmnfld

        return normals_nonmnfld, sdfs_nonmnfld


def koch_line(start, end, factor):
    """
    Segments a line to Koch line, creating fractals.


    :param tuple start:  (x, y) coordinates of the starting point
    :param tuple end: (x, y) coordinates of the end point
    :param float factor: the multiple of sixty degrees to rotate
    :returns tuple: tuple of all points of segmentation
    """

    # coordinates of the start
    x1, y1 = start[0], start[1]

    # coordinates of the end
    x2, y2 = end[0], end[1]

    # the length of the line
    l = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # first point: same as the start
    a = (x1, y1)

    # second point: one third in each direction from the first point
    b = (x1 + (x2 - x1) / 3., y1 + (y2 - y1) / 3.)

    # third point: rotation for multiple of 60 degrees
    c = (b[0] + l / 3. * np.cos(factor * np.pi / 3.), b[1] + l / 3. * np.sin(factor * np.pi / 3.))

    # fourth point: two thirds in each direction from the first point
    d = (x1 + 2. * (x2 - x1) / 3., y1 + 2. * (y2 - y1) / 3.)

    # the last point
    e = end

    return {'a': a, 'b': b, 'c': c, 'd': d, 'e': e, 'factor': factor}


def koch_snowflake(degree, s=1.0):
    """Generates all lines for a Koch Snowflake with a given degree.
    code from: https://github.com/IlievskiV/Amusive-Blogging-N-Coding/blob/master/Visualizations/snowflake.ipynb
    :param int degree: how deep to go in the branching process
    :param float s: the length of the initial equilateral triangle
    :returns list: list of all lines that form the snowflake
    """
    # all lines of the snowflake
    lines = []

    # we rotate in multiples of 60 degrees
    sixty_degrees = np.pi / 3.

    # vertices of the initial equilateral triangle
    A = (0., 0.)
    B = (s, 0.)
    C = (s * np.cos(sixty_degrees), s * np.sin(sixty_degrees))

    # set the initial lines
    if degree == 0:
        lines.append(koch_line(A, B, 0))
        lines.append(koch_line(B, C, 2))
        lines.append(koch_line(C, A, 4))
    else:
        lines.append(koch_line(A, B, 5))
        lines.append(koch_line(B, C, 1))
        lines.append(koch_line(C, A, 3))

    for i in range(1, degree):
        # every lines produce 4 more lines
        for _ in range(3 * 4 ** (i - 1)):
            line = lines.pop(0)
            factor = line['factor']

            lines.append(koch_line(line['a'], line['b'], factor % 6))  # a to b
            lines.append(koch_line(line['b'], line['c'], (factor - 1) % 6))  # b to c
            lines.append(koch_line(line['c'], line['d'], (factor + 1) % 6))  # d to c
            lines.append(koch_line(line['d'], line['e'], factor % 6))  # d to e

    return lines


def get_koch_points(degree, s=1.0):
    lines = koch_snowflake(degree, s=s)
    points = []
    for line in lines:
        for key in line.keys():
            if not key == 'factor' and not key == 'e':
                points.append(line[key])
    points = np.array(points) - np.array([s/2, (s/2)*np.tan(np.pi/6)])
    points = np.flipud(points) #reorder the points clockwise
    # plt.plot(points[:, 0], points[:, 1])
    # plt.show()
    return points


def get2D_dataset(*args, shape_type='snowflake', line_sample_type='uniform', model_scale=1):

    if shape_type == 'circle':
        out_shape = Circle(*args, radius=0.5, cx=0.0, cy=0.0)
    elif shape_type == 'double_circle':
        out_shape = double_Circle(*args, radius=0.2)
    elif shape_type == 'noisy_circle':
        out_shape = Noisy_Circle(*args, radius=0.8 * model_scale)
    elif shape_type == 'L':
        length = model_scale * 0.5
        out_shape = Polygon(*args, vertices=[[0., 0.], [length, 0.], [length, -length],
                            [-length, -length], [-length, length], [0, length]], line_sample_type=line_sample_type)
    elif shape_type == 'L_thin':
        length = model_scale * 0.5
        out_shape = Polygon(*args, vertices=[[0., 0.], [length, 0.], [length, -length/50],
                            [-length/50, -length/50], [-length/50, length], [0, length]], line_sample_type=line_sample_type)
    elif shape_type == 'noisy_L':
        length = model_scale * 0.5
        out_shape = Noisy_Polygon(*args, vertices=[[0., 0.], [length, 0.], [length, -length],
                                  [-length, -length], [-length, length], [0, length]], line_sample_type=line_sample_type)
    elif shape_type == 'square':
        length = model_scale * 0.5
        out_shape = Polygon(*args, vertices=[[-length, length], [length, length], [length, -length],
                                             [-length, -length]], line_sample_type=line_sample_type)
    elif shape_type == 'noisy_square':
        length = model_scale * 0.5
        out_shape = Noisy_Polygon(*args, vertices=[[-length, length], [length, length], [length, -length],
                                                   [-length, -length]], line_sample_type=line_sample_type)
    elif shape_type == 'hexagon':
        length = model_scale * 0.5
        out_shape = Polygon(*args, vertices=[[-length, length], [0, length * 1.5], [length, length], [length, -length],
                                             [0, -length * 1.5], [-length, -length]],
                            line_sample_type=line_sample_type)
    elif shape_type == 'noisy_hexagon':
        length = model_scale * 0.5
        out_shape = Noisy_Polygon(*args, vertices=[[-length, length], [0, length * 1.5], [length, length], [length, -length],
                                                   [0, -length * 1.5], [-length, -length]],
                                  line_sample_type=line_sample_type)
    elif shape_type == 'snowflake':
        vertices = get_koch_points(degree=2, s=0.8 * model_scale)
        out_shape = Polygon(*args, vertices=vertices, line_sample_type=line_sample_type)
    elif shape_type == 'random_sf':
        vertices = get_koch_points(degree=2, s=0.8 * model_scale)
        for i in range(vertices.shape[0]):
            vertices[i] += np.random.rand(2) * 0.08 - 0.04
        out_shape = Polygon(*args, vertices=vertices, line_sample_type=line_sample_type)
    elif shape_type == 'noisy_snowflake':
        vertices = get_koch_points(degree=2, s=0.8 * model_scale)
        out_shape = Noisy_Polygon(*args, vertices=vertices, line_sample_type=line_sample_type)
    else:
        raise Warning("Unsupportaed shape")

    return out_shape




