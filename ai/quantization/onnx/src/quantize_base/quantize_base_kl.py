# -*- coding:utf-8 -*- #
# @Author  :wang dian
import numba
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


def _smooth_distribution(p, eps=0.0001):
    """Given a discrete distribution (may have not been normalized to 1),
    smooth it by replacing zeros with eps multiplied by a scaling factor and taking the
    corresponding amount off the non-zero values.
    Ref: http://hanj.cs.illinois.edu/cs412/bk3/KL-divergence.pdf
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist



def _find_scale_by_kl(arr, quantized_dtype='uint8', num_bins=4001, num_quantized_bins=255):
    """Given a tensor, find the optimal threshold for quantizing it.
    The reference distribution is `q`, and the candidate distribution is `p`.
    `q` is a truncated version of the original distribution.

    Ref:
    http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    """
    assert isinstance(arr, np.ndarray)
    assert stats is not None, "scipy needs to be installed for utilizing kl calibration during quantization"
    min_val = np.min(arr)
    max_val = np.max(arr)
    th = max(abs(min_val), abs(max_val))

    if min_val >= 0 and quantized_dtype in ['uint8']:
        # We need to move negative bins to positive bins to fit uint8 range.
        num_quantized_bins = num_quantized_bins * 2 + 1

    hist, hist_edges = np.histogram(arr, bins=num_bins, range=(-th, th))
    zero_bin_idx = num_bins // 2
    num_half_quantized_bins = num_quantized_bins // 2

    thresholds = np.zeros(num_bins // 2 + 1 - num_quantized_bins // 2)
    divergence = np.zeros_like(thresholds)
    quantized_bins = np.zeros(num_quantized_bins, dtype=np.int32)
    # i means the number of bins on half axis excluding the zero bin.
    for i in range(num_quantized_bins // 2, num_bins // 2 + 1):
        p_bin_idx_start = zero_bin_idx - i
        p_bin_idx_stop = zero_bin_idx + i + 1
        thresholds[i - num_half_quantized_bins] = hist_edges[p_bin_idx_stop]
        sliced_nd_hist = hist[p_bin_idx_start:p_bin_idx_stop]

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        assert p.size % 2 == 1
        assert p.size >= num_quantized_bins
        # put left outlier count in p[0]
        left_outlier_count = np.sum(hist[0:p_bin_idx_start])
        p[0] += left_outlier_count
        # put right outlier count in p[-1]
        right_outlier_count = np.sum(hist[p_bin_idx_stop:])
        p[-1] += right_outlier_count
        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int32)

        # calculate how many bins should be merged to generate quantized distribution q
        num_merged_bins = sliced_nd_hist.size // num_quantized_bins
        # merge hist into num_quantized_bins bins
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[num_quantized_bins * num_merged_bins:].sum()
        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float32)
        for j in range(num_quantized_bins):
            start = j * num_merged_bins
            if j == num_quantized_bins - 1:
                stop = len(is_nonzeros)
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        q[p == 0] = 0
        p = _smooth_distribution(p)
        # There is a chance that q is an invalid probability distribution.
        try:
            q = _smooth_distribution(q)
        except ValueError:
            divergence[i - num_half_quantized_bins] = float("inf")
        divergence[i - num_half_quantized_bins] = stats.entropy(p, q)

    min_divergence_idx = np.argmin(divergence)
    opt_th = thresholds[min_divergence_idx]
    print("T {}".format(opt_th))
    return opt_th, divergence[min_divergence_idx]


def ThresholdLayerIntputs_2d(data, bitwidth):
    data = data.flatten()

    positive_data = data[np.where(data >= 0)]
    negative_data = data[np.where(data < 0)]

    positive_threshold = 0.0
    negative_threshold = 0.0

    total_min_loss = float("inf")

    positive_min_loss = 0
    negative_min_loss = 0
    positive_tmp_threshold = 0
    negative_tmp_threshold = 0

    if len(positive_data) > 0:
        # positive_tmp_threshold, positive_min_loss = ComputeThreshold(positive_data, 127, 'sqrt')

        positive_tmp_threshold, positive_min_loss = _find_scale_by_kl(data)
        print("positive_min_loss : {}".format(positive_min_loss))
    if len(negative_data) > 0:
        # negative_tmp_threshold, negative_min_loss = ComputeThreshold(negative_data, 2, 'sqrt')
        negative_tmp_threshold, negative_min_loss = _find_scale_by_kl(negative_data)
        print("negative_min_loss : {}".format(negative_min_loss))

    tmp_min_loss = pow(positive_min_loss, 2) + pow(negative_min_loss, 2)
    if tmp_min_loss < total_min_loss:
        total_min_loss = tmp_min_loss
        positive_threshold = positive_tmp_threshold
        negative_threshold = negative_tmp_threshold
    print("negative_threshold : {} positive_threshold: {}".format(negative_threshold * -1, positive_threshold))
    return negative_threshold * -1, positive_threshold


def ThresholdLayerOutputs(data, bitwidth):
    data = data.flatten()
    data_min = np.min(data)
    if data_min >= 0:
        positive_threshold, positive_min_loss = _find_scale_by_kl(data)
        return 0, positive_threshold
    else:
        positive_threshold, positive_min_loss = _find_scale_by_kl(np.abs(data))
        return -1*positive_threshold, positive_threshold

