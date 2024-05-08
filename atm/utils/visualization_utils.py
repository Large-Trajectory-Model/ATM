import os
import numpy as np
from moviepy.editor import ImageSequenceClip


def make_grid(array, ncol=5, padding=0, pad_value=120):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if np.max(array) < 2.:
        array = array * 255.
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    if N % ncol > 0:
        res = ncol - N % ncol
        array = np.concatenate([array, np.ones([res, H, W, C])])
        N = array.shape[0]
    nrow = N // ncol
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]],
                     constant_values=pad_value, mode='constant')
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]],
                             constant_values=pad_value, mode='constant')
            row = np.hstack([row, cur_img])
        idx += 1
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
    return grid_img.astype(np.float32)


def save_numpy_as_video(array, filename, fps=20, extension='mp4'):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    """
    import cv2

    if np.max(array) <= 2.:
        array *= 255.
    array = array.astype(np.uint8)
    # ensure that the file has the .mp4 extension
    fname, _ = os.path.splitext(filename)
    filename = fname + f'.{extension}'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip without interpalation
    clip = ImageSequenceClip(list(array), fps=fps)
    clip.write_videofile(filename, fps=fps, logger=None)
    return clip
