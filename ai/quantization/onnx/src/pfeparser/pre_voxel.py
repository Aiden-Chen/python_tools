# -*- coding:utf-8 -*- #
import numpy as np
from collections import defaultdict
import config as const


def points_to_voxel_3d_np(points, voxels, coors, num_points_per_voxel, coor_to_voxelidx,
                          voxel_size, coors_range, max_points, max_voxels):
    '''coor_to_voxelidx存储了voxel坐标对应的idx编号（从0开始编号）的映射，值初始为-1,注意key时voxel坐标组成的tuple'''
    NDim = 3
    num_features = 4
    ndim_minus_1 = NDim - 1
    voxel_num = 0
    N, num_features = points.shape[0], points.shape[1]  # 点数
    grid_size = np.zeros(3, )
    coor = np.ones((3,), dtype=np.int32)
    # 获取点云体素的三个维度大小
    for i in range(NDim):
        grid_size[i] = round((coors_range[NDim + i] - coors_range[i]) / voxel_size[i])
    for i in range(N):
        failed = False
        for j in range(NDim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])  # 某个点某个维度的大小所落的voxel坐标
            # 若超出点云立方体素，则忽略
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[ndim_minus_1 - j] = c  # 把xyz的三个维度的voxel坐标按zyx次序放置 coor存储了当前点的坐标zyx次序的voxel坐标
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]  # voxelidx存储了当前点的坐标zyx次序的voxel坐标对应的编号
        if voxelidx == -1:  # 初始就是-1,即还没有点落入此voxel
            voxelidx = voxel_num
            if voxel_num >= max_voxels:  # 超出最大保留体素格数
                break
            voxel_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            for k in range(NDim):
                coors[voxelidx, k] = coor[k]  # coors存储voxel坐标（20000,3）
        num = num_points_per_voxel[voxelidx]  # num_points_per_voxel存储每个体素格的点数（20000，）
        if num < max_points:  #
            for k in range(num_features):  # 将当前点的特征放在voxels对应编号是voxelidx的体素格中第num个的数据位置
                voxels[voxelidx, num, k] = points[i, k]
            num_points_per_voxel[voxelidx] += 1  # 将当前点所在的体素格中的点数加1
    # 重新将coor_to_voxelidx的val置为原位-1
    for i in range(voxel_num):
        coor_to_voxelidx[coors[i, 0], coors[i, 1], coors[i, 2]] = -1
    return voxel_num


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    coor_to_voxelidx,
                    max_points=35,
                    max_voxels=20000):
    """convert 3d points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 0.8ms(~6k voxels)
    with c++ and 3.2ghz cpu.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        coor_to_voxelidx: int array. used as a dense map.
        max_points: int. indicate maximum points contained in a voxel.
        max_voxels: int. indicate maximum voxels this function create.
            for voxelnet, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor. zyx format.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    voxelmap_shape = voxelmap_shape[::-1]
    num_points_per_voxel = np.zeros(shape=(max_voxels,), dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)
    voxel_num = points_to_voxel_3d_np(
        points, voxels, coors, num_points_per_voxel, coor_to_voxelidx,
        voxel_size.tolist(), coors_range.tolist(), max_points, max_voxels)
    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    return voxels, coors, num_points_per_voxel


class VoxelGenerator:
    def __init__(self,
                 voxel_size,
                 point_cloud_range,
                 max_num_points,
                 max_voxels=const.MAX_VOXELS):
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        # [0, -40, -3, 70.4, 40, 1]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        voxelmap_shape = tuple(np.round(grid_size).astype(np.int32).tolist())
        voxelmap_shape = voxelmap_shape[::-1]  # XYZ-->ZYX

        self._coor_to_voxelidx = np.full(voxelmap_shape, -1, dtype=np.int32)
        print(self._coor_to_voxelidx.shape)
        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points, max_voxels=None):
        res = points_to_voxel(
            points, self._voxel_size, self._point_cloud_range, self._coor_to_voxelidx,
            self._max_num_points, max_voxels or self._max_voxels)
        return res

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points

    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size


def build(points, voxel_size, point_clound_range, max_points_per_voxel):
    """Builds a tensor dictionary based on the InputReader config.

    Args:
        input_reader_config: A input_reader_pb2.InputReader object.

    Returns:
        A tensor dict based on the input_reader_config.

    Raises:
        ValueError: On invalid input reader proto.
        ValueError: If no input paths are specified.
    """
    #     if not isinstance(voxel_config, (voxel_generator_pb2.VoxelGenerator)):
    #         raise ValueError('input_reader_config not of type '
    #                          'input_reader_pb2.InputReader.')

    voxel_config = {"voxel_size": voxel_size, "point_cloud_range": point_clound_range,
                    "max_number_of_points_per_voxel": max_points_per_voxel}
    voxel_generator = VoxelGenerator(
        voxel_size=list(voxel_config["voxel_size"]),
        point_cloud_range=list(voxel_config["point_cloud_range"]),
        max_num_points=voxel_config["max_number_of_points_per_voxel"],
        max_voxels=const.MAX_VOXELS)
    return voxel_generator.generate(points, max_voxels=const.MAX_VOXELS)


def merge_second_batch(batch_list):
    example_merged = defaultdict(list)
    for example in batch_list:
        for k, v in example.items():
            example_merged[k].append(v)
    ret = {}
    for key, elems in example_merged.items():
        if key in [
            'voxels', 'num_points', 'num_gt', 'voxel_labels', 'gt_names', 'gt_classes', 'gt_boxes'
        ]:
            ret[key] = np.concatenate(elems, axis=0)
        elif key == 'metadata':
            ret[key] = elems
        elif key == "calib":
            ret[key] = {}
            for elem in elems:
                for k1, v1 in elem.items():
                    if k1 not in ret[key]:
                        ret[key][k1] = [v1]
                    else:
                        ret[key][k1].append(v1)
            for k1, v1 in ret[key].items():
                ret[key][k1] = np.stack(v1, axis=0)
        elif key == 'coordinates':
            coors = []
            for i, coor in enumerate(elems):
                coor_pad = np.pad(
                    coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                coors.append(coor_pad)
            ret[key] = np.concatenate(coors, axis=0)
        elif key == 'metrics':
            ret[key] = elems
        else:
            ret[key] = np.stack(elems, axis=0)
    return ret


def preprocess(bin_file, voxel_size, point_clound_range, max_points_per_voxel):
    example = {}
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    voxel_generator_res = build(points, voxel_size, point_clound_range, max_points_per_voxel)
    voxels, coors, num_points_per_voxel = voxel_generator_res
    print('voxels.shape:', voxels.shape)
    print('coors shape:', coors.shape)
    print('num_points_per_voxel:', num_points_per_voxel.shape)
    example = {
        'voxels': voxels,
        'num_points': num_points_per_voxel,
        'coordinates': coors,
    }
    example = merge_second_batch([example])
    return example
