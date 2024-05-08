import torch
from torch import nn
from einops import rearrange

class TrackPatchEmbed(nn.Module):
    def __init__(self,
                 num_track_ts,
                 num_track_ids,
                 patch_size,
                 in_dim,
                 embed_dim):
        super().__init__()
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.patch_size = patch_size
        self.in_dim = in_dim
        self.embed_dim = embed_dim

        assert self.num_track_ts % self.patch_size == 0, "num_track_ts must be divisible by patch_size"
        self.num_patches_per_track = self.num_track_ts // self.patch_size
        self.num_patches = self.num_patches_per_track * self.num_track_ids

        self.conv = nn.Conv1d(in_dim, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, tracks):
        """
        tracks: (B, T, N, in_dim)

        embed the tracks into patches. make sure to reshape into (B, N, T, out_dim) at the end
        """
        b, t, n, c = tracks.shape
        tracks = rearrange(tracks, 'b t n c -> (b n) c t')
        patches = self.conv(tracks)
        patches = rearrange(patches, '(b n) c t -> b t n c', b=b, n=n)

        return patches