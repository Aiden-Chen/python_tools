# -*- coding:utf-8 -*- #
import numpy as np


def quantize_weight_2d(weight, bitwise):
    weight_scale = []
    weight_zero_point = []
    kernel = np.shape(weight)[0]
    for k in range(kernel):
        weight_c = weight[k, :, :, :]
        max_weight_c = np.max(weight_c)
        min_weight_c = np.min(weight_c)
        max_weight_val = np.max([max_weight_c, -1 * min_weight_c])
        # weight_scale_float = max_weight_val / ((pow(2, bitwise) - 1) / 2) # 127.5
        weight_scale_float = max_weight_val / (pow(2, bitwise - 1)-1)  # 127
        weight_scale.append(weight_scale_float)
        weight_zero_point.append(0)
    return weight_scale, weight_zero_point


def quantize_weight_tarnpose2d(weight, bitwise):
    weight_scale = []
    weight_zero_point = []
    kernel = np.shape(weight)[1]
    for k in range(kernel):
        weight_c = weight[:, k, :, :]
        max_weight_c = np.max(weight_c)
        min_weight_c = np.min(weight_c)
        max_weight_val = np.max([max_weight_c, -1 * min_weight_c])
        # weight_scale_float = max_weight_val / ((pow(2, bitwise) - 1) / 2)
        weight_scale_float = max_weight_val / (pow(2, bitwise - 1)-1)  # 127
        weight_scale.append(weight_scale_float)
        weight_zero_point.append(0)
    return weight_scale, weight_zero_point


def quantize_weight_3d(weight, bitwise):
    weight_scale = []
    weight_zero_point = []
    kernel = np.shape(weight)[-1]
    for k in range(kernel):
        weight_c = weight[:, :, :, :, k]
        max_weight_c = np.max(weight_c)
        min_weight_c = np.min(weight_c)
        max_weight_val = np.max([max_weight_c, -1 * min_weight_c])
        # weight_scale_float = max_weight_val / ((pow(2, bitwise) - 1) / 2) # 127.5
        weight_scale_float = max_weight_val / (pow(2, bitwise - 1)-1)  # 127
        weight_scale.append(weight_scale_float)
        weight_zero_point.append(0)
    return weight_scale, weight_zero_point

def quanti_weight_channel_symmetry_quantitative(weights, quanti_bits):
    """
    量化权重数据、权重的数据shape为k,c,h,w
    量化方式为逐通道、对称量化
    :param weights: 需要量化的权重
    :param quanti_bis: 权重量化的位宽
    :return: 返回权重的量化系数、量化零点、最大值、最小值、
    """
    weight_scale = []
    weight_zero_point = []
    max_value = pow(2, quanti_bits - 1) - 1

    for i in range(len(weights[0])):
        weights_k = weights[0][i]
        weight_k_max = np.max(np.abs(weights_k))
        weight_scale_float = weight_k_max / max_value
        weight_scale.append(weight_scale_float)
        weight_zero_point.append(0)
        del weights_k, weight_k_max, weight_scale_float

    del max_value
    return weight_scale, weight_zero_point, np.max(weights), np.min(weights)
