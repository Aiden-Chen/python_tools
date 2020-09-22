# from second.snowlake_sdk.3_model_quantize.src.config.config import Size_Dict
import numpy as np
import config as const


def get_network_param():
    point_clound_range = np.array(const.POINT_CLOUND_RANGE)
    input_shape = (point_clound_range[3:] - point_clound_range[:3]) / const.VOXEL_SIZE
    input_shape = input_shape[::-1].astype(np.int32)  # [40, 1600, 1408]
    output_shape = input_shape / const.DOWNSAMPLE_RATE
    output_shape[0] = 1
    output_size = int(output_shape[1] * output_shape[2])
    anchor_stride = (input_shape[1:] / (output_shape[1:] - 1) * const.VOXEL_SIZE[:-1]).tolist()[::-1]
    anchor_stride.extend([0.0])

    input_shape = input_shape.tolist()
    output_shape = output_shape.tolist()
    print('input_shape:{}'.format(input_shape))  # [40, 1600, 1408]
    print('output_shape:{}'.format(output_shape))  # [1.0, 200.0, 176.0]
    print('output_size:{}'.format(output_size))    # 200.0 * 176.0
    print('anchor_stride:{}'.format(anchor_stride))
    return input_shape, output_shape, output_size, anchor_stride


def create_anchors_3d_stride(feature_size,
                             anchor_strides,
                             sizes=[1.6, 3.9, 1.56],
                             anchor_offsets=[0, -20, -1],  # [0.2, -39.8, -1.78],
                             rotations=[0, 1.57],  # np.pi / 2
                             dtype=np.float32):
    """
    Args:
        feature_size: list [D, H, W](zyx)
        sizes: [N, 3] list of list or array, size of anchors, xyz

    Returns:
        anchors: [*feature_size, num_sizes, num_rots, 7] tensor.
    """
    # almost 2x faster than v1
    x_stride, y_stride, z_stride = anchor_strides
    x_offset, y_offset, z_offset = anchor_offsets
    z_centers = np.arange(feature_size[0], dtype=dtype)
    y_centers = np.arange(feature_size[1], dtype=dtype)
    x_centers = np.arange(feature_size[2], dtype=dtype)
    z_centers = z_centers * z_stride + z_offset
    y_centers = y_centers * y_stride + y_offset
    x_centers = x_centers * x_stride + x_offset
    sizes = np.reshape(np.array(sizes, dtype=dtype), [-1, 3])
    rotations = np.array(rotations, dtype=dtype)
    rets = np.meshgrid(
        x_centers, y_centers, z_centers, rotations, indexing='ij')
    tile_shape = [1] * 5
    tile_shape[-2] = int(sizes.shape[0])
    for i in range(len(rets)):
        rets[i] = np.tile(rets[i][..., np.newaxis, :], tile_shape)
        rets[i] = rets[i][..., np.newaxis]  # for concat
    sizes = np.reshape(sizes, [1, 1, 1, -1, 1, 3])
    tile_size_shape = list(rets[0].shape)
    tile_size_shape[3] = 1
    sizes = np.tile(sizes, tile_size_shape)
    rets.insert(3, sizes)
    ret = np.concatenate(rets, axis=-1)
    return np.transpose(ret, [2, 1, 0, 3, 4, 5])


# 验证一下逻辑
# def get_anchors_4_class(output_shape, output_size, stride, size, offset):
#     car_res = create_anchors_3d_stride(output_shape,
#                                        anchor_strides=stride,
#                                        sizes=size['Car'],
#                                        anchor_offsets=offset['Car'])  # -1-(4.5~5-1.73)
#     car_res_0 = car_res[:, :, :, :, 0, :].reshape(-1, output_size, 7)
#     car_res_1 = car_res[:, :, :, :, 1, :].reshape(-1, output_size, 7)
# #
#     cyclist_res = create_anchors_3d_stride(output_shape,
#                                            anchor_strides=stride,
#                                            sizes=size['Cyclist'],
#                                            anchor_offsets=offset['Cyclist'])  # -0.6-(4.5~5-1.73)
#     cyclist_res_0 = cyclist_res[:, :, :, :, 0, :].reshape(-1, output_size, 7)
#     cyclist_res_1 = cyclist_res[:, :, :, :, 1, :].reshape(-1, output_size, 7)
#
#     pedestrian_res = create_anchors_3d_stride(output_shape,
#                                               anchor_strides=stride,
#                                               sizes=size['Pedestrian'],
#                                               anchor_offsets=offset['Pedestrian'])
#     pedestrian_res_0 = pedestrian_res[:, :, :, :, 0, :].reshape(-1, output_size, 7)
#     pedestrian_res_1 = pedestrian_res[:, :, :, :, 1, :].reshape(-1, output_size, 7)
#
#     van_res = create_anchors_3d_stride(output_shape,
#                                        anchor_strides=stride,
#                                        sizes=size['Van'],
#                                        anchor_offsets=offset['Van'])
#     van_res_0 = van_res[:, :, :, :, 0, :].reshape(-1, output_size, 7)
#     van_res_1 = van_res[:, :, :, :, 1, :].reshape(-1, output_size, 7)
#
#     res = np.concatenate(
#         [car_res_0, car_res_1, cyclist_res_0, cyclist_res_1, pedestrian_res_0, pedestrian_res_1, van_res_0, van_res_1],
#         axis=1)
#     return res

def get_anchors_4_class(output_shape, output_size, stride, size, offset):

    # size and offset key have same numbers , and have same name
    # get all same name(dict of key)
    class_name = size.keys()
    # for return to use
    result = []

    for name in class_name:
        res = create_anchors_3d_stride(output_shape,
                                           anchor_strides = stride,
                                           sizes = size[name],
                                           anchor_offsets=offset[name])  # -1-(4.5~5-1.73)
        res_0 = res[:, :, :, :, 0, :].reshape(-1, output_size, 7)
        res_1 = res[:, :, :, :, 1, :].reshape(-1, output_size, 7)
        result.append(res_0)
        result.append(res_1)

    ret = np.concatenate(result, axis = 1)
    return ret


# if __name__ == '__main__':
def get_anchors():
    input_shape, output_shape, output_size, anchor_stride = get_network_param()
    # size_dict = {
    #     'Car': [1.6, 3.9, 1.56],
    #     'Cyclist': [0.6, 1.76, 1.73],
    #     'Pedestrian': [0.6, 0.8, 1.73],
    #     'Van': [1.87103749, 5.02808195, 2.20964255]
    # }
    # offset_dict = {
    #     'Car': [0, -40, -1],
    #     'Cyclist': [0, -40, -1],
    #     'Pedestrian': [0, -40, -1],
    #     'Van': [0, -40, -1]
    # }
    size_dict = const.Size_Dict
    offset_dict = const.Offset_Dict
    anchor_res = get_anchors_4_class(output_shape, output_size, anchor_stride, size_dict, offset_dict)
    with open(const.ANCHORS_PATH, "wb") as f:
        f.write(anchor_res)
