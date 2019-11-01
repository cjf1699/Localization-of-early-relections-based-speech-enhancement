from __future__ import print_function
from __future__ import division
from numpy import *
import numpy as np
from scipy.special import hankel2
import config as c

# REF: 3D Sound Field Recording with Higher Order Ambisonics -
# Objective Measurements and Validation of Spherical Microphone


def spheric_harmonics_function(a, b, azimuth, elevation):
    """
    spheric harmonics function.
    :param a: int, function order, a <= 4
    :param b: int, function number
    :param azimuth: double or numpy array
    :param elevation: double or numpy array  是仰角，不是倾斜角
    :return: double or numpy array
    """
    if a == 0 and b == +0: return sqrt(1 / 4 / pi) * ones_like(azimuth)
    if a == 1 and b == +0: return sqrt(3 / 4 / pi) * sin(elevation)
    if a == 1 and b == -1: return sqrt(3 / 8 / pi) * cos(elevation) * sin(azimuth) * sqrt(2)
    if a == 1 and b == +1: return sqrt(3 / 8 / pi) * cos(elevation) * cos(azimuth) * sqrt(2)
    if a == 2 and b == +0: return sqrt(5 / 16 / pi) * (3 * sin(elevation) ** 2 - 1)
    if a == 2 and b == -1: return sqrt(15 / 8 / pi) * cos(elevation) * sin(elevation) * sin(azimuth) * sqrt(2)
    if a == 2 and b == +1: return sqrt(15 / 8 / pi) * cos(elevation) * sin(elevation) * cos(azimuth) * sqrt(2)
    if a == 2 and b == -2: return sqrt(15 / 32 / pi) * cos(elevation) ** 2 * sin(2 * azimuth) * sqrt(2)
    if a == 2 and b == +2: return sqrt(15 / 32 / pi) * cos(elevation) ** 2 * cos(2 * azimuth) * sqrt(2)
    if a == 3 and b == +0: return sqrt(7 / 16 / pi) * (5 * (sin(elevation)) ** 3 - 3 * sin(elevation))
    if a == 3 and b == -1: return sqrt(21 / 64 / pi) * cos(elevation) * (5 * (sin(elevation)) ** 2 - 1) * sin(
        azimuth) * sqrt(2)
    if a == 3 and b == +1: return sqrt(21 / 64 / pi) * cos(elevation) * (5 * (sin(elevation)) ** 2 - 1) * cos(
        azimuth) * sqrt(2)
    if a == 3 and b == -2: return sqrt(105 / 32 / pi) * (cos(elevation)) ** 2 * sin(elevation) * sin(
        2 * azimuth) * sqrt(2)
    if a == 3 and b == +2: return sqrt(105 / 32 / pi) * (cos(elevation)) ** 2 * sin(elevation) * cos(
        2 * azimuth) * sqrt(2)
    if a == 3 and b == -3: return sqrt(35 / 64 / pi) * (cos(elevation)) ** 3 * sin(3 * azimuth) * sqrt(2)
    if a == 3 and b == +3: return sqrt(35 / 64 / pi) * (cos(elevation)) ** 3 * cos(3 * azimuth) * sqrt(2)
    if a == 4 and b == +0: return sqrt(9 / 4 / pi) * (1 / 8) * (
            35 * (sin(elevation)) ** 4 - 30 * (sin(elevation)) ** 2 + 3)
    if a == 4 and b == -1: return sqrt(9 / 160 / pi) * cos(elevation) * (
            35 * sin(elevation) ** 3 - 15 * sin(elevation)) * sin(azimuth)
    if a == 4 and b == +1: return sqrt(9 / 160 / pi) * cos(elevation) * (
            35 * sin(elevation) ** 3 - 15 * sin(elevation)) * cos(azimuth)
    if a == 4 and b == -2: return sqrt(45 / 64 / pi) * cos(elevation) ** 2 * (7 * (sin(elevation)) ** 2 - 1) * sin(
        2 * azimuth)
    if a == 4 and b == +2: return sqrt(45 / 64 / pi) * cos(elevation) ** 2 * (7 * (sin(elevation)) ** 2 - 1) * cos(
        2 * azimuth)
    if a == 4 and b == -3: return sqrt(315 / 64 / pi) * cos(elevation) ** 3 * sin(elevation) * sin(3 * azimuth) * sqrt(
        2)
    if a == 4 and b == +3: return sqrt(315 / 64 / pi) * cos(elevation) ** 3 * sin(elevation) * cos(3 * azimuth) * sqrt(
        2)
    if a == 4 and b == -4: return (3 / 16) * sqrt(35 / 2 / pi) * (cos(elevation)) ** 4 * sin(4 * azimuth) * sqrt(2)
    if a == 4 and b == +4: return (3 / 16) * sqrt(35 / 2 / pi) * (cos(elevation)) ** 4 * cos(4 * azimuth) * sqrt(2)

    raise ValueError("a=%s, b=%s. a, b should be int and abs(b) < a." % (a, b))


def spheric_harmonics_list(azimuth, elevation):
    sh_list = np.zeros((c.hoa_num, azimuth.size))
    hoa_index = 0
    for a in range(c.hoa_order+1):
        for b in range(-a, a + 1):
            sh_list[hoa_index, :] = spheric_harmonics_function(a, b, azimuth, elevation)
            hoa_index += 1
    return sh_list


def get_y():
    # reference paper, Y was defined near equation (23)
    # check pass.
    y = zeros((c.n_chan, c.hoa_num), dtype='complex')
    index = 0
    for a in range(c.hoa_order + 1):
        for b in range(-a, a + 1):
            y[:, index] = spheric_harmonics_function(a, b, c.mic_position[:, 1], pi / 2 - c.mic_position[:, 0])
            index += 1
    return y


def get_bn(freq, order):
    kr = 2 * np.pi * freq * c.array_radius / c.speed_of_sound + (freq == 0) * np.finfo(float).eps
    _t = sqrt(np.pi / (2 * kr)) * (order / kr * hankel2(order + 1 / 2, kr) - hankel2(order + 3 / 2, kr))
    bn = 1j ** (order - 1) / (kr ** 2 * _t)
    return bn


# 计算论文（30）式的函数
def get_B(freq):
    B = zeros(c.hoa_num, dtype='complex')
    index = 0
    for a in range(c.hoa_order + 1):
        for b in range(-a, a + 1):
            B[index] = get_bn(freq, a)
            index += 1
    return np.diag(B)


def get_y2(az, el):
    """
    For now, we use az and el as scalars    -----  2019.5.17
    :param az: azimuth. double or ndarray
    :param el: elevation. double or ndarray
    :return: sphere harmonic functions for the given (az, el)   (c.hoa_num, n_az)
    """
    n_az = az.size
    # assert n_az == n_el  # i.e. az and el must come in pairs

    y = zeros((c.hoa_num, n_az), dtype='float64')
    index = 0
    for a in range(c.hoa_order + 1):
        for b in range(-a, a + 1):
            y[index, :] = spheric_harmonics_function(a, b, az, el)
            index += 1
    return y


def get_eq(freq, order, param=0.18):
    # check pass.
    bn = get_bn(freq, order)
    out = np.conjugate(bn) / ((abs(bn)) ** 2 + param ** 2)
    return out


def get_eq_matrix(freq):
    # check pass.
    eq_matrix = np.zeros((c.hoa_order + 1) ** 2, dtype=complex)
    for i in range(c.hoa_order + 1):
        for j in range(2 * i + 1):
            eq_matrix[i ** 2 + j] = get_eq(freq, i)
    return np.diag(eq_matrix)


def get_encoder():
    y = get_y()
    e_matrix = linalg.pinv(y)
    freq_array = linspace(0, c.fs / 2, c.fft_point // 2 + 1)
    encoder = zeros((freq_array.size, c.hoa_num, c.n_chan), dtype=complex)

    # f = open('cond.txt', 'w')

    for freq_index, freq in enumerate(freq_array):
        eq_matrix = get_eq_matrix(freq)

        # diag_w = np.linalg.inv(eq_matrix)
        # TT = y.dot(diag_w)
        # f.write('freq:'+str(freq_index)+' '+str(np.linalg.cond(TT.conj().T.dot(TT)))+'\n')

        encoder[freq_index, :, :] = eq_matrix.dot(e_matrix)

    return encoder


def get_valid_freq_array_and_index():
    freq_array = np.linspace(0, c.fs // 2, c.fft_point // 2 + 1)
    valid_freq_index = (freq_array > c.min_freq) * (freq_array < c.max_freq)
    return valid_freq_index, freq_array[valid_freq_index]


def get_decoder():
    spheric_harmonics = np.zeros((c.hoa_num, c.scan_num, c.el_num))
    for az_index, az in enumerate(c.az_array):
        for el_index, el in enumerate(c.el_array):
            spheric_harmonics[:, az_index, el_index] = spheric_harmonics_list(az, np.pi / 2 - el).ravel()
    return spheric_harmonics


def get_weight_vector():
    _, valid_freq_array = get_valid_freq_array_and_index()
    kr_array = 2 * np.pi * valid_freq_array / c.speed_of_sound * c.array_radius
    n = 3  # warning: magic number
    weight_vector = 1 / kr_array ** 2 / (np.sqrt(np.pi / (2 * kr_array)) * (
                n / kr_array * hankel2(n + 1 / 2, kr_array) - hankel2(n + 3 / 2, kr_array)))
    return np.abs(weight_vector)


def get_p_vector(freq, azi, el):
    """

    :param freq: scalar in Hz
    :param azi: double or ndarray.
    :param el: double or ndarray
    :return: p_vector for each given pair (az, el) in azi and el. Refer to paper for more info.
    """
    n_az, n_el = azi.size, el.size
    # assert n_az == n_el  # i.e. az and el must come in pairs
    p_vector = np.zeros((c.hoa_num, n_az), dtype="complex")
    Y = get_y2(azi, el)
    index = 0
    for i in range(c.hoa_order+1):
        b_n = get_bn(freq, i)
        for j in range(-i, i+1):
            p_vector[index] = b_n * Y[index, :]
            index += 1
    return p_vector


if __name__ == '__main__':
    for i in range(5):
        for j in range(-i, i+1):
            print(spheric_harmonics_function(i, j, 2*np.pi/3, 0))

