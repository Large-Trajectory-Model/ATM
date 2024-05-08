import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
import matplotlib.pyplot as plt
from matplotlib import cm
import cv2


class ImageUnNormalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(ImageUnNormalize, self).__init__()
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        if self.mean.ndim == 1:
            self.mean = self.mean.view(-1, 1, 1)
        if self.std.ndim == 1:
            self.std = self.std.view(-1, 1, 1)

    def forward(self, tensor):
        self.mean = self.mean.to(tensor.device)
        self.std = self.std.to(tensor.device)
        return tensor * self.std + self.mean


def get_track_displacement(tracks):
    """
    track: (B, T, N, 2)
    return: (B, N)

    caluculate the displacement of each track by taking the magnitude of the
    difference between each timestep, then summing over the timesteps.
    """
    b, t, c, n = tracks.shape
    diff_tracks = torch.diff(tracks, dim=1)
    mag_tracks = torch.norm(diff_tracks, dim=-1)
    disp_tracks = torch.sum(mag_tracks, dim=1)
    return disp_tracks


def sample_grid(n, device="cuda", dtype=torch.float32, left=(0.1, 0.1), right=(0.9, 0.9)):
    # sample nxn points as a grid
    u = torch.linspace(left[0], right[0], n, device=device, dtype=dtype)
    v = torch.linspace(left[1], right[1], n, device=device, dtype=dtype)
    u, v = torch.meshgrid(u, v)
    u = u.reshape(-1)
    v = v.reshape(-1)
    points = torch.stack([u, v], dim=-1)
    return points


def sample_double_grid(n, device="cuda", dtype=torch.float32,):
    points1 = sample_grid(n, device, dtype, left=(0.05, 0.05), right=(0.85, 0.85))
    points2 = sample_grid(n, device, dtype, left=(0.15, 0.15), right=(0.95, 0.95))
    points = torch.cat([points1, points2], dim=0)
    return points


def sample_tracks_nearest_to_grids(tracks, vis, num_samples):
    """
    Sample the tracks whose first points are nearest to the grids
    Args:
        tracks: (track_len n 2)
        vis: (track_len n)
        num_samples: number of tracks to sample
    Returns:
        (track_len num_samples 2)
    """
    assert num_samples == 32
    reference_grid_points = sample_double_grid(n=4, device="cpu")  # (32, 2)

    first_points = tracks[0]  # (n, 2)
    dist = torch.norm(first_points[:, None, :] - reference_grid_points[None, :, :], dim=-1)  # (n, 32)
    nearest_idx = torch.argmin(dist, dim=0)  # (32,)
    nearest_tracks = tracks[:, nearest_idx, :]  # (track_len, 32, 2)
    nearest_vis = vis[:, nearest_idx]  # (track_len, 32)
    return nearest_tracks, nearest_vis


def sample_tracks(tracks, num_samples=16, uniform_ratio=0.25, vis=None, motion=False, h=None):
    """
    tracks: (T, N, 2)
    num_samples: int, number of samples to take
    uniform_ratio: float, ratio of samples to take uniformly vs. according to displacement
    return: (T, num_samples, 2)

    sample num_samples tracks from the tracks tensor, using both uniform sampling and sampling according
    to the track displacement.
    """

    t, n, c = tracks.shape

    if motion:
        mask = (tracks > 0) & (tracks < h)
        mask = mask.all(dim=-1) # if any of u, v is out of bounds, then it's false
        mask = mask.all(dim=0) # if any of the points in the track is out of bounds, then it's false

        mask = repeat(mask, 'n -> t n', t=t)
        tracks = tracks[mask]
        tracks = tracks.reshape(t, -1, c)

        if vis is not None:
            t, n = vis.shape
            vis = vis[mask]
            vis = vis.reshape(t, -1)

    num_uniform = int(num_samples * uniform_ratio)
    num_disp = num_samples - num_uniform

    uniform_idx = torch.randint(0, n, (num_uniform,))

    if num_disp == 0:
        idx = uniform_idx
    else:
        disp = get_track_displacement(tracks[None])[0]
        threshold = disp.min() + (disp.max() - disp.min()) * 0.1
        disp[disp < threshold] = 0
        disp[disp >= threshold] = 1
        disp_idx = torch.multinomial(disp, num_disp, replacement=True)

        idx = torch.cat([uniform_idx, disp_idx], dim=-1)

    sampled_tracks = tracks[:, idx]
    if vis is not None:
        t, n = vis.shape
        sampled_vis = vis[:, idx]

        return sampled_tracks, sampled_vis

    return sampled_tracks

def sample_tracks_visible_first(tracks, vis, num_samples=16):
    """
    Only sample points which are visible on the initial frame
    tracks: (T, N, 2)
    vis: (T, N)
    num_samples: int, number of samples to take
    return: (T, num_samples, 2)

    sample num_samples tracks from the tracks tensor, using both uniform sampling and sampling according
    to the track displacement.
    """
    t, n, c = tracks.shape

    vis_idx = torch.where(vis[0] >0)[0]

    idx = torch.randint(0, len(vis_idx), (num_samples,))

    sampled_tracks = tracks[:, vis_idx[idx]]
    sampled_vis = vis[:, vis_idx[idx]]
    return sampled_tracks, sampled_vis


def tracks_to_binary_img(tracks, img_size):
    """
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    return: (B, T, C, H, W)
    """
    from einops import repeat
    B, T, N, C = tracks.shape
    generation_size = 128
    H, W = generation_size, generation_size

    tracks = tracks * generation_size
    u, v = tracks[:, :, :, 0].long(), tracks[:, :, :, 1].long()
    u = torch.clamp(u, 0, W - 1)
    v = torch.clamp(v, 0, H - 1)
    uv = u + v * W

    img = torch.zeros(B, T, H * W).to(tracks.device)
    img = img.scatter(2, uv, 1).view(B, T, H, W)

    # img size is b x t x h x w
    img = repeat(img, 'b t h w -> (b t) h w')[:, None, :, :]
    import torch.nn.functional as F
    # Generate 5x5 gaussian kernel
    kernel = [[0.003765, 0.015019, 0.023792, 0.015019, 0.003765],
              [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
              [0.023792, 0.094907, 0.150342, 0.094907, 0.023792],
              [0.015019, 0.059912, 0.094907, 0.059912, 0.015019],
              [0.003765, 0.015019, 0.023792, 0.015019, 0.003765]]
    kernel /= np.max(kernel)
    kernel = torch.FloatTensor(kernel)[None, None, :, :].to(tracks.device)
    img = F.conv2d(img, kernel, padding=2)[:, 0, :, :]
    img = rearrange(img, '(b t) h w -> b t h w', b=B)
    if generation_size != img_size:
        img = F.interpolate(img, size=(img_size, img_size), mode="bicubic")
    img = torch.clamp(img, 0, 1)
    img = torch.where(img < 0.05, torch.tensor(0.0), img)

    img = repeat(img, 'b t h w -> b t c h w', c=3)

    assert torch.max(img) <= 1
    return img


def tracks_to_video(tracks, img_size):
    """
    tracks: (B, T, N, 2), where each track is a sequence of (u, v) coordinates; u is width, v is height
    return: (B, C, H, W)
    """
    B, T, N, _ = tracks.shape
    binary_vid = tracks_to_binary_img(tracks, img_size=img_size).float()  # b, t, c, h, w
    binary_vid[:, :, 0] = binary_vid[:, :, 1]
    binary_vid[:, :, 2] = binary_vid[:, :, 1]

    # Get blue to purple cmap
    cmap = plt.get_cmap('coolwarm')
    cmap = cmap(1 / np.arange(T))[:T, :3][::-1]
    binary_vid = binary_vid.clone()

    for l in range(T):
        # interpolate betweeen blue and red
        binary_vid[:, l, 0] = binary_vid[:, l, 0] * cmap[l, 0] * 255
        binary_vid[:, l, 1] = binary_vid[:, l, 1] * cmap[l, 1] * 255
        binary_vid[:, l, 2] = binary_vid[:, l, 2] * cmap[l, 2] * 255
    # Overwride from the last frame
    track_vid = torch.sum(binary_vid, dim=1)
    track_vid[track_vid > 255] = 255
    return track_vid

def combine_track_and_img(track: torch.Tensor, vid: np.ndarray):
    """
    track: [B, T, N, 2]
    vid: [B, C, H, W]
    return: (B, C, H, W)
    """
    img_size = vid.shape[-1]
    track_video = tracks_to_video(track, img_size)  # B 3 H W
    track_video = track_video.detach().cpu().numpy()
    vid = vid.copy().astype(np.float32)
    vid[track_video > 0] = track_video[track_video > 0]
    return vid.astype(np.uint8)


def draw_traj_on_images(tracks: torch.Tensor, images: np.ndarray, show_dots=False):
    """
    tracks: [B, T, N, 2]
    images: [B, C, H, W]
    Returns: [B, C, H, W]
    """
    b, c, h, w = images.shape
    assert c == 3

    images_back = images.astype(np.uint8).copy()
    images_back = rearrange(images_back, "b c h w -> b h w c")
    images_back = images_back.copy()

    tracks[:, :, :, 0] = torch.clamp(tracks[:, :, :, 0] * h, 0, h-1)
    tracks[:, :, :, 1] = torch.clamp(tracks[:, :, :, 1] * w, 0, w-1)

    color_map = cm.get_cmap("cool")
    linewidth = max(int(5 * h / 512), 1)

    result_images = []
    for traj_set, img in zip(tracks, images_back):
        traj_len  = traj_set.shape[0]
        for traj_idx in range(traj_set.shape[1]):
            traj = traj_set[:, traj_idx]  # (T, 2)

            for s in range(traj_len - 1):
                color = np.array(color_map((s) / max(1, traj_len - 2))[:3]) * 255  # rgb
                # print(int(traj[s, 0]), int(traj[s, 1]), int(traj[s + 1, 0]), int(traj[s + 1, 1]))

                cv2.line(img, pt1=(int(traj[s, 0]), int(traj[s, 1])), pt2=(int(traj[s + 1, 0]), int(traj[s + 1, 1])),
                    color=color,
                    thickness=linewidth,
                    lineType=cv2.LINE_AA)
                if show_dots:
                    cv2.circle(img, (traj[s, 0], traj[s, 1]), linewidth, color, -1)
        result_images.append(img)

    result_images = np.stack(result_images, dtype=np.uint8)
    result_images = rearrange(result_images, "b h w c -> b c h w")
    return result_images


def sample_from_mask(mask, num_samples=16, replace=False):
    """
    mask: (H, W, 1) np
    num_samples: int, number of samples to take
    return: (num_samples, 2), where this is the (u, v) coordinates of the sampled pixels in the mask
    """

    # write the code according to the docstring above
    h, w, c = mask.shape
    mask = rearrange(mask, 'h w c -> (h w) c')

    idxs = np.where(mask == 255)[0]
    if len(idxs) == 0:
        # return random samples from the image
        idxs = np.arange(h*w)
        np.random.shuffle(idxs)

    if num_samples == -1:
        num_samples = len(idxs)
    if not replace:
        num_samples = min(num_samples, len(idxs))
    idxs = np.random.choice(idxs, num_samples, replace=replace)

    # split into x and y
    u = idxs % w
    v = idxs // w

    return np.stack([u, v], axis=-1)
