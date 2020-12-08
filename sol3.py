from scipy import signal
import numpy as np
import skimage.color as sk
from skimage import io
import matplotlib.pyplot as plt


def blur(im, filter_size):
    vec_sum = np.sum(get_filter_vec(filter_size))
    conv_lines = signal.convolve2d(im, get_filter_vec(filter_size) / vec_sum, 'same')
    t_filter = np.transpose(get_filter_vec(filter_size))
    conv_cols = signal.convolve2d(conv_lines, t_filter / vec_sum, 'same')
    return conv_cols


def reduce(im, filter_size):
    blured_im = blur(im, filter_size)
    return blured_im[::2, ::2]


def get_filter_vec(filter_size):
    kernel = np.array([[1, 1]])
    res = signal.convolve2d(kernel, kernel)
    for i in range(filter_size - 2):
        res = signal.convolve2d(res, kernel)
    return res


def build_gaussian_pyramid(im, max_levels, filter_size):
    filter_vec = get_filter_vec(filter_size)
    gaussian_array = [im]
    for i in range(int(max_levels - 1)):
        im = reduce(im, filter_size)
        gaussian_array.append(im)
    return gaussian_array, filter_vec


def example():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    print(arr[::2, ::2])


def read_image(filename, representation):
    im = io.imread(filename)
    im_float = im.astype(np.float64)
    im_float /= 255

    if representation == 1:
        im_g = sk.rgb2gray(im_float)
        return im_g
    elif representation == 2:
        return im_float


if __name__ == '__main__':
    # print(get_filter_vec(4))
    # example()
    x = read_image("monkey.jpg", 1)
    m, n = build_gaussian_pyramid(x, 5, 3)
    plt.imshow(m[3], cmap="gray")
    plt.show()
    print(n)
