import numpy as np
from atm.utils.visualization_utils import make_grid, save_numpy_as_video


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid_video_from_numpy(video_array, ncol, output_name='./output.mp4', speedup=1, padding=5, **kwargs):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=padding)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)
