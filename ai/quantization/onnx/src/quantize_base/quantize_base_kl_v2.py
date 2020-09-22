# -*- coding:utf-8 -*- #
import collections
import configparser
import logging
import numpy as np
from numba import njit

config = configparser.ConfigParser(delimiters=":")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s:%(levelname)s: %(message)s")


def write_ini_file(file_name, layer_name, parm_list, val_list):
    """
    Write the quantization parameters to a file with the suffix ini
    :param file_name: a file name with the suffix ini to be write
    :param layer_name: layer name to be write
    :param parm_list: a list of parameters name
    :param val_list: a list of values corresponding to the parameters name order in parm_list
    :return:
    """
    dict_parm_val = collections.OrderedDict()
    for keys, values in zip(parm_list, val_list):
        for k, v in zip(keys, values):
            dict_parm_val[k] = v
    config[layer_name] = dict_parm_val
    with open(file_name, 'w') as configfile:
        config.write(configfile, space_around_delimiters=False)


def Float2Fixed2Float(data, bitwidth, threshold, f):
    if np.isclose(threshold, 0.0):
        threshold = np.zeros_like(threshold)
    scaling_factor = threshold / (pow(2, bitwidth - 1) - 1)
    orig = np.array(data)
    data = np.clip(data, -threshold, threshold)
    if threshold != 0:
        data /= scaling_factor
        data = f(data)
        data *= scaling_factor
    error = np.sum(np.square(orig - data))
    return data


# @njit
def CdfMeasure(x, y, measure_name):
    if measure_name == 'Kullback-Leibler-J':
        return np.sum((x - y) * np.log2(x / y))
    else:
        return CdfMeasure(x, y, 'Kullback-Leibler-J')


# @njit
def ComputeThreshold(data, bitwidth, bins):
    abs_data = np.abs(data)
    mn = 0
    mx = abs_data.max()
    # print('Min: ', mn, ', Max: ', mx)
    zed = np.float32(0.0)
    if np.isclose(mx, zed):
        th_layer_out = zed
        sf_layer_out = zed
        print('Mean : th_layer_out: ', th_layer_out, ', sf_layer_out: ', sf_layer_out)
        return th_layer_out

    hist, bin_edges = np.histogram(abs_data, bins, range=(mn, mx), density=True)
    hist = hist / np.sum(hist)
    cumsum = np.cumsum(hist)
    n = pow(2, bitwidth)
    threshold = []
    scaling_factor = []
    d = []
    # print('n: ', n, ', len(bin_edges): ', len(bin_edges))

    if n + 1 > len(bin_edges) - 1:
        th_layer_out = bin_edges[-1]
        # sf_layer_out = th_layer_out / (pow(2, bitwidth - 1) - 1)
        # print('Mean : th_layer_out: ', th_layer_out, ', sf_layer_out: ', sf_layer_out)
        return th_layer_out

    for i in range(n + 1, len(bin_edges), 1):
        threshold_tmp = (i + 0.5) * (bin_edges[1] - bin_edges[0])
        threshold = np.concatenate((threshold, [threshold_tmp]))
        scaling_factor_tmp = threshold_tmp / (pow(2, bitwidth - 1) - 1)
        scaling_factor = np.concatenate((scaling_factor, [scaling_factor_tmp]))
        p = np.copy(cumsum)
        p[(i - 1):] = 1
        x = np.linspace(0.0, 1.0, n)
        xp = np.linspace(0.0, 1.0, i)
        fp = p[:i]
        p_interp = np.interp(x, xp, fp)
        x = np.linspace(0.0, 1.0, i)
        xp = np.linspace(0.0, 1.0, n)
        fp = p_interp
        q_interp = np.interp(x, xp, fp)
        q = np.copy(p)
        q[:i] = q_interp
        d_tmp = CdfMeasure(cumsum, q, 'Kullback-Leibler-J')
        d = np.concatenate((d, [d_tmp]))

    th_layer_out = threshold[np.argmin(d)]

    # mean = (mx - mn) / 2.0
    # max_th = mean + (np.argmin(d) + 0.5) * (mx - mn) / bins
    # min_th = mean - (np.argmin(d) + 0.5) * (mx - mn) / bins

    sf_layer_out = scaling_factor[np.argmin(d)]
    logging.info('Mean : th_layer_out:{} , sf_layer_out:{} '.format(th_layer_out, sf_layer_out))
    # assert type(th_layer_out) == np.float64
    return np.float32(th_layer_out)


def ThresholdLayerInputs(data, bitwidth):
    threshold = np.max(np.abs(data))
    return threshold


def ThresholdWeights(data, bitwidth):
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    return threshold


def ThresholdBiases(data, bitwidth):
    threshold = np.max(np.abs(data), axis=tuple(range(1, data.ndim)))
    return threshold


def ThresholdLayerOutputs(data, bitwidth):
    min_val = np.min(data)
    threshold = ComputeThreshold(data.flatten(), bitwidth, 'sqrt')
    if min_val >= 0:
        return 0, threshold
    return -threshold, threshold


def QuantizeBlob(data, bitwidth):
    _, threshold = ThresholdLayerOutputs(data, bitwidth)
    return (
        Float2Fixed2Float(data, bitwidth, threshold, np.round), threshold)


def QuantizeThresholdBlob(data, bitwidth, threshold):
    assert type(threshold) in [np.float32, np.float64], 'Theshold is not a scalar'
    return Float2Fixed2Float(data, bitwidth, threshold, np.round)


def QuantizeWeights(threshold, bitwidth, data, mode='caffe'):
    if mode == 'tf':
        data = data.transpose(2, 3, 1, 0)
    assert data.shape[0] == threshold.shape[0], 'Threshold shape does not match weight data shape'
    for i in range(len(threshold)):
        data[i] = Float2Fixed2Float(data[i], bitwidth, threshold[i], np.round)

    if mode == 'tf':
        data = data.transpose(3.2, 0, 1)
    return data
