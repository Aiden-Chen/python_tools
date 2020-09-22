# -*- coding:utf-8 -*- #
import numpy as np
import pandas as pd
import heapq


def get_max_min_2d(min_list_all, max_list_all, n=100):
    min_list = pd.Series(min_list_all)
    data_count = min_list.value_counts(bins=2, normalize=True, sort=False)
    print(data_count)
    for k, v in data_count.items():
        if v > 0.7:
            num_thresh = str(k).split(',')[-1].split(']')[0]  # k.right即可代替，取区间的右值
            min_list = np.array(min_list[min_list.astype(np.float32) <= float(num_thresh)])
    if len(min_list) > n:
        min_list = np.array(min_list)
        min_list_index = heapq.nsmallest(n, range(len(min_list)), min_list.take)
        min_list = min_list[min_list_index]
    elif len(min_list) == 0:
        min_list = min_list_all
    min_input_all = np.mean(min_list)

    max_list = pd.Series(max_list_all)
    data_count = max_list.value_counts(bins=2, normalize=True, sort=False)
    print(data_count)
    for k, v in data_count.items():
        if v > 0.7:
            num_thresh = str(k).split(',')[0].split('(')[1]  # k.left代替即可
            max_list = np.array(max_list[max_list.astype(np.float32) >= float(num_thresh)])  # 理论上应该取大于
    if len(max_list) > n:
        max_list = np.array(max_list)
        max_list_index = heapq.nlargest(n, range(len(max_list)), max_list.take)
        max_list = max_list[max_list_index]
    elif len(max_list) == 0:
        max_list = max_list_all
    max_input_all = np.mean(max_list)
    return min_input_all, max_input_all


def get_max_min_3d(min_list_all, max_list_all, n=100):
    min_list = pd.Series(min_list_all)
    data_count = min_list.value_counts(bins=2, normalize=True, sort=False)
    print(data_count)
    for k, v in data_count.items():
        if v > 0.7:
            num_thresh = str(k).split(',')[-1].split(']')[0]  # k.right即可代替，取区间的右值
            min_list = np.array(min_list[min_list.astype(np.float32) <= float(num_thresh)])
    if len(min_list) > n:
        min_list = np.array(min_list)
        min_list_index = heapq.nsmallest(n, range(len(min_list)), min_list.take)
        min_list = min_list[min_list_index]
    min_input_all = np.min(min_list)

    max_list = pd.Series(max_list_all)
    data_count = max_list.value_counts(bins=2, normalize=True, sort=False)
    print(data_count)
    for k, v in data_count.items():
        if v > 0.7:
            num_thresh = str(k).split(',')[0].split('(')[1]  # k.left代替即可
            max_list = np.array(max_list[max_list.astype(np.float32) >= float(num_thresh)])  # 理论上应该取大于
    if len(max_list) > n:
        max_list = np.array(max_list)
        max_list_index = heapq.nlargest(n, range(len(max_list)), max_list.take)
        max_list = max_list[max_list_index]
    max_input_all = np.max(max_list)
    return min_input_all, max_input_all
