"""
Author: Jude Park, Jan 18th 2020.
"""


import numpy as np
import matplotlib.pyplot as plt


def get_angles(pos, i, d_model):
    """
    positional encoding mathematical expression.
    :param pos: position of each given token
    :param i: 2i or 2i+1, each condition is for sin, cos function.
    :param d_model: dimension of given model
    :return:
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    """
    extract positional encoding.
    :param position:
    :param d_model:
    :return:
    """
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # mapping rule for 2i, 2i+1
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return angle_rads


d_model = 128
max_seq_len = 64

assert d_model % 2 == 0

pos_encoding = positional_encoding(max_seq_len, d_model)

plt.pcolormesh(pos_encoding, cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, d_model))
plt.ylabel('Position')
plt.colorbar()
plt.show()
