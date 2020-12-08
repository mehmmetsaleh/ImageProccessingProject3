from scipy import signal
import numpy as np


def build_gaussian_pyramid(im, max_levels, filter_size):
    pass


def blur(im, filter_size):
    conv_lines = signal.convolve2d(im, get_filter_vec(filter_size), 'same')
    t_filter = np.transpose(get_filter_vec(filter_size))
    conv_cols = signal.convolve2d(conv_lines, t_filter, 'same')


def reduce(im, filter_size):
    pass


def get_filter_vec(filter_size):
    kernel = np.array([[1, 1]])
    res = signal.convolve2d(kernel, kernel)
    for i in range(filter_size - 2):
        res = signal.convolve2d(res, kernel)
    return res


def example():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr[::2, ::2])


if __name__ == '__main__':
    print(get_filter_vec(4))
    # example()
