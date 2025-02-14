import numpy as np
from torch.utils.data import Dataset


class SDFDataset_1D(Dataset):
    def __init__(self, size=100, num_samples=2 ** 18, batchsize=2 ** 18, online=False, clip_sdf=None):
        super().__init__()
        a = -1
        b = 1
        self.bbox = np.array([[-2.1], [2.1]]).astype(np.float32)
        self.points_mnfld = np.asarray([[a], [b]])
        self.size = size
        self.num_samples = num_samples
        self.batchsize = batchsize
        self.online = online
        self.sampling_std = abs(b-a)
        # self.clip_sdf = clip_sdf
        self.exist_normals = True

        if not self.online:
            self.points_nonmnfld = self.bbox[0] + (self.bbox[1] - self.bbox[0]) * np.linspace(0, 1, num=self.num_samples)[:, None]
            # self.points_nonmnfld = self.bbox[0] + (self.bbox[1] - self.bbox[0]) * np.random.rand(self.num_samples, 1)

            self.points_near = np.concatenate([np.random.normal(a, self.sampling_std, size=(self.num_samples//2, 1)),
                                               np.random.normal(b, self.sampling_std, size=(self.num_samples//2, 1))
                                               ], axis=0)
            self.points_near = np.where(np.abs(self.points_near) > self.bbox[1],
                                        np.sign(self.points_near) * self.bbox[1] * 0.95,
                                        self.points_near)
            self.sdfs_near = -np.sign(np.abs(self.points_near-(a+b)/2)-np.abs((b-a)/2)) * \
                             np.minimum(np.abs(self.points_near-a), np.abs(self.points_near-b))
            self.normals_near = -np.sign(self.points_near - (a + b) / 2)

            self.sdfs_mnfld = np.zeros((2, 1))
            self.normals_mnfld = np.array([[-1], [1]])
            self.sdfs_nonmnfld = -np.sign(np.abs(self.points_nonmnfld-(a+b)/2)-np.abs((b-a)/2)) * np.minimum(np.abs(self.points_nonmnfld-a),
                                                                                                 np.abs(self.points_nonmnfld-b))
            self.normals_nonmnfld = -np.sign(self.points_nonmnfld - (a+b)/2)
            self.points_target_near = np.where(self.points_near > (a+b)/2, b, a)
            self.points_target_nonmnfld = np.where(self.points_nonmnfld > (a + b) / 2, b, a)
            self.sigmas1 = np.abs(self.sdfs_near)

    def __len__(self):
        return self.size

    def __getitem__(self, _):
        if self.online:
            a = self.points_mnfld[0]
            b = self.points_mnfld[1]
            points_mnfld = self.points_mnfld.astype(np.float32)
            points_nonmnfld = (self.bbox[0] + (self.bbox[1] - self.bbox[0]) *
                               np.linspace(0, 1, num=self.num_samples))[:, None].astype(np.float32)
            # points_nonmnfld = (self.bbox[0] + (self.bbox[1] - self.bbox[0]) *
            #                    np.random.rand(self.num_samples, 1)).astype(np.float32)

            points_near = np.concatenate([np.random.normal(a, self.sampling_std, size=(self.num_samples//2, 1)),
                                          np.random.normal(b, self.sampling_std, size=(self.num_samples//2, 1))
                                          ], axis=0)
            points_near = np.where(np.abs(points_near) > self.bbox[1],
                                   np.sign(points_near) * self.bbox[1] * 0.95, points_near)
            sdfs_near = -np.sign(np.abs(points_near - (a + b) / 2) - np.abs((b - a) / 2)) * \
                        np.minimum(np.abs(points_near - a), np.abs(points_near - b))
            normals_near = -np.sign(points_near - (a + b) / 2)
            points_target_near = np.where(points_near > (a + b) / 2, b, a)
            sigmas1 = np.abs(sdfs_near)

            sdfs_mnfld = np.zeros((2, 1))
            normals_mnfld = np.array([[-1], [1]])
            sdfs_nonmnfld = -np.sign(np.abs(points_nonmnfld - (a + b) / 2) - np.abs((b - a) / 2)) * np.minimum(
                np.abs(points_nonmnfld - a), np.abs(points_nonmnfld - b)).astype(np.float32)
            normals_nonmnfld = -np.sign(points_nonmnfld - (a + b) / 2).astype(np.float32)
            points_target_nonmnfld = np.where(points_nonmnfld > (a + b) / 2, b, a)
        else:
            points_mnfld = self.points_mnfld.astype(np.float32)
            choose_array = np.random.choice(self.num_samples, self.batchsize, replace=False)
            points_nonmnfld = self.points_nonmnfld[choose_array, :].astype(np.float32)
            sdfs_mnfld = np.zeros((2, 1))
            normals_mnfld = np.array([[-1], [1]])
            sdfs_nonmnfld = self.sdfs_nonmnfld[choose_array, :].astype(np.float32)
            normals_nonmnfld = self.normals_nonmnfld[choose_array, :].astype(np.float32)
            points_near = self.points_near[choose_array, :].astype(np.float32)
            sdfs_near = self.sdfs_near[choose_array, :].astype(np.float32)
            normals_near = self.normals_near[choose_array, :].astype(np.float32)
            points_target_near = self.points_target_near[choose_array, :].astype(np.float32)
            points_target_nonmnfld = self.points_target_nonmnfld[choose_array, :].astype(np.float32)
            sigmas1 = self.sigmas1

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
            'vertices': self.points_mnfld,
            'exist_normals': self.exist_normals,
            'sigmas1': sigmas1,
            'points_target_near': points_target_near,
            'points_target_nonmnfld': points_target_nonmnfld
        }
        return results