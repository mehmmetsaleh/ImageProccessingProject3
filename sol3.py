from scipy import signal
import numpy as np
import skimage.color as sk
from skimage import io
import matplotlib.pyplot as plt
import os


def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)


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
    for i in range(max_levels - 1):
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


def linear_stretch_pyramid(pyr, levels):
    for i in range(levels):
        pyr[i] = (pyr[i] - pyr[i].min()) / (pyr[i].max() - pyr[i].min())
    return pyr


def render_pyramid_helper(pyr, levels):
    new_pyr = [pyr[0]]
    for i in range(1, levels):
        new_arr = np.zeros((pyr[0].shape[0], pyr[i].shape[1]))
        new_arr[:pyr[i].shape[0]:, ::] = pyr[i]
        new_pyr.append(new_arr)
    return new_pyr


def render_pyramid(pyr, levels):
    stretched = linear_stretch_pyramid(pyr, levels)
    res = render_pyramid_helper(stretched, levels)
    img = np.hstack(res)
    return img


def display_pyramid(pyr, levels):
    img = render_pyramid(pyr, levels)
    plt.imshow(img, cmap="gray")
    plt.show()


def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
    l1, filter_vec1 = build_laplacian_pyramid(im1, max_levels, filter_size_im)
    l2, filter_vec2 = build_laplacian_pyramid(im2, max_levels, filter_size_im)
    gaussian_mask, mask_filter_vec = build_gaussian_pyramid(mask.astype(np.float64), max_levels, filter_size_mask)
    l_out = []
    for k in range(max_levels):
        lk = gaussian_mask[k] * l1[k] + (1.0 - gaussian_mask[k]) * l2[k]
        if lk.shape[0] < 16 or lk.shape[1] < 16:
            break
        l_out.append(lk)

    blended_im = laplacian_to_image(l_out, get_filter_vec(filter_size_im), [1] * len(l_out))
    blended_clipped_im = np.clip(blended_im, 0, 1)
    return blended_clipped_im


def blending_example1():
    im1 = read_image(relpath("externals/me.jpg"), 2)
    im2 = read_image(relpath("externals/wick.jpg"), 2)
    mask = read_image(relpath("externals/wick_mask.jpg"), 1)
    blend_im = np.zeros(im1.shape)
    blend_im[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 5, 3, 3)
    blend_im[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 5, 3, 3)
    blend_im[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 5, 3, 3)

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()

    return im1, im2, mask.astype(np.bool), blend_im


def blending_example2():
    im1 = read_image(relpath("externals/wing_final.jpg"), 2)
    im2 = read_image(relpath("externals/giraffe_final.jpg"), 2)
    mask = read_image(relpath("externals/giraffe_mask.jpg"), 1)
    blend_im = np.zeros(im1.shape)
    blend_im[:, :, 0] = pyramid_blending(im1[:, :, 0], im2[:, :, 0], mask, 5, 3, 3)
    blend_im[:, :, 1] = pyramid_blending(im1[:, :, 1], im2[:, :, 1], mask, 5, 3, 3)
    blend_im[:, :, 2] = pyramid_blending(im1[:, :, 2], im2[:, :, 2], mask, 5, 3, 3)

    plt.subplot(2, 2, 1)
    plt.imshow(im1)
    plt.subplot(2, 2, 2)
    plt.imshow(im2)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap="gray")
    plt.subplot(2, 2, 4)
    plt.imshow(blend_im)
    plt.show()

    return im1, im2, mask.astype(np.bool), blend_im


def read_image(filename, representation):
    im = io.imread(filename)
    im_float = im.astype(np.float64)
    im_float /= 255

    if representation == 1:
        im_g = sk.rgb2gray(im_float)
        return im_g
    elif representation == 2:
        return im_float
