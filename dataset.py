import sys

import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset


def construct_coordinate(sample_points_num, theta):
    x = np.linspace(-1, 1, sample_points_num)
    y = np.linspace(-1, 1, sample_points_num)
    X, Y = np.meshgrid(x, y, indexing='ij')  # 注意indexing方式 -------> 方向
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                                [np.sin(theta), np.cos(theta)]])

    # 使用numpy的向量化操作提取每一个坐标点对
    coords = np.column_stack((X.ravel(), Y.ravel())).T
    new_coords = np.matmul(rotation_matrix, coords)
    new_coords = new_coords.T.reshape(sample_points_num, sample_points_num, 2)
    return new_coords


class SinogramDataset(Dataset):
    def __init__(self, sample_points_num, views, sample_N=10, sino_path=None, train=True):
        if sino_path is None and train is True:
            print("Sinogram path not given!")
            sys.exit(1)
        self.train = train
        if self.train:
            self.sinogram = sitk.GetArrayFromImage(sitk.ReadImage(sino_path))
        self.sample_points_num = sample_points_num
        self.views = views
        self.sample_N = sample_N
        rays = []
        intensities = []
        angles = np.linspace(0, np.pi, self.views, endpoint=False)
        for i in range(views):
            rays.append(construct_coordinate(sample_points_num, angles[i]))
            if self.train:
                intensities.append(self.sinogram[i, :])
        self.rays = np.array(rays)
        self.intensities = np.array(intensities)
        pass

    def __len__(self):
        return len(self.rays)

    def __getitem__(self, index):
        ray = self.rays[index]
        if self.train:
            intensity = self.intensities[index]
            sample_idx = np.random.choice(len(ray), size=self.sample_N, replace=False)
            ray_sample = ray[sample_idx]
            intensity_sample = intensity[sample_idx]
            return ray_sample, intensity_sample
        else:
            return ray


if __name__ == '__main__':
    path = "../SCOPE/data/gt_sino.nii"
    train_data = SinogramDataset(sample_points_num=367, views=60, sino_path=path)
