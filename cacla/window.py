"""
Class for smoothing learning curves.
"""

import numpy as np
import queue


def window_convolution(a, H):
    """
    Helper method to average the last H values.
    :param a: array, size n
    :param H: integer
    :return: array, size n-H
    """
    v = []
    sum_H = 0
    q = queue.Queue(H)
    for i in range(len(a)):
        if q.full():
            sum_H -= q.get()
            q.put(a[i])
            sum_H += a[i]
            v.append(sum_H)
        else:
            q.put(a[i])
            sum_H += a[i]
    return np.array(v)/H