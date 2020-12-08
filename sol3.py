from scipy import signal
import numpy as np
import skimage.color as sk
from skimage import io
import matplotlib.pyplot as plt


def blur_reduce(im, filter_size):
    vec_sum = np.sum(get_filter_vec(filter_size))
    conv_lines = signal.convolve2d(im, get_filter_vec(filter_size) / vec_sum, 'same')
    t_filter = np.transpose(get_filter_vec(filter_size))
    conv_cols = signal.convolve2d(conv_lines, t_filter / vec_sum, 'same')
    return conv_cols


def reduce(im, filter_size):
    blurred_im = blur_reduce(im, filter_size)
    return blurred_im[::2, ::2]


def get_filter_vec(filter_size):
    kernel = np.array([[1, 1]])
    res = signal.convolve2d(kernel, kernel)
    for i in range(filter_size - 3):
        res = signal.convolve2d(res, kernel)
    return res


def build_gaussian_pyramid(im, max_levels, filter_size):
    filter_vec = get_filter_vec(filter_size)
    gaussian_array = [im]
    for i in range(int(max_levels - 1)):
        if im.shape[0] < 16 or im.shape[1] < 16:
            break
        im = reduce(im, filter_size)
        gaussian_array.append(im)
    return gaussian_array, filter_vec


def expand(im, filter_size):
    padded_im = np.zeros((2 * im.shape[0], 2 * im.shape[1]))
    padded_im[::2, ::2] = im
    blurred_im = blur_expand(padded_im, filter_size)
    return blurred_im


def blur_expand(im, filter_size):
    normalizing_val = 2 / np.sum(get_filter_vec(filter_size))
    conv_lines = signal.convolve2d(im, get_filter_vec(filter_size) * normalizing_val, 'same')
    t_filter = np.transpose(get_filter_vec(filter_size))
    conv_cols = signal.convolve2d(conv_lines, t_filter * normalizing_val, 'same')
    return conv_cols


def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    laplacian_arr = [pyr[0] - expand(pyr[1], filter_size)]
    for i in range(1, max_levels - 1):
        laplacian_arr.append(pyr[i] - expand(pyr[i + 1], filter_size))
    laplacian_arr.append(pyr[-1])
    return laplacian_arr, filter_vec


def expand_all(lpyr, filter_size):
    for i in range(1, len(lpyr)):
        for j in range(i):
            lpyr[i] = expand(lpyr[i], filter_size)
    return lpyr


def laplacian_to_image(lpyr, filter_vec, coeff):
    same_size_lpyr = expand_all(lpyr, len(filter_vec))
    res = same_size_lpyr[0]
    for i in range(1, len(same_size_lpyr)):
        res += coeff[i] * same_size_lpyr[i]
    return res


def example():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # print(arr[::2, ::2])
    new_arr = np.zeros((2 * arr.shape[0], 2 * arr.shape[1]))
    new_arr[::2, ::2] = arr
    print(new_arr)


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
    x = read_image("city.jpg", 1)
    # m, n = build_gaussian_pyramid(x, 5, 3)
    # plt.imshow(m[4], cmap="gray")
    # plt.show()
    m2, n2 = build_laplacian_pyramid(x, 5, 3)
    # plt.imshow(m2[0], cmap="gray")
    # plt.show()
    # print(m2[0].shape)
    # print(m2[1].shape)
    # print(m2[2].shape)
    plt.subplot(2,2,1)
    plt.imshow(x,cmap="gray")
    img = laplacian_to_image(m2, get_filter_vec(3), [1, 1, 1, 1, 1])
    plt.subplot(2,2,2)
    plt.imshow(img, cmap="gray")
    plt.show()