import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from einops import rearrange, repeat
from timm.models.vision_transformer import PatchEmbed
from torch import nn

from atm.utils.flow_utils import ImageUnNormalize, tracks_to_video
from atm.utils.pos_embed_utils import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from atm.policy.vilt_modules.language_modules import *
from .track_patch_embed import TrackPatchEmbed
from .transformer import Transformer

class TrackTransformer(nn.Module):
    """
    flow video model using a BERT transformer

    dim: int, dimension of the model
    depth: int, number of layers
    heads: int, number of heads
    dim_head: int, dimension of each head
    attn_dropout: float, dropout for attention layers
    ff_dropout: float, dropout for feedforward layers
    """

    def __init__(self,
                 transformer_cfg,
                 track_cfg,
                 vid_cfg,
                 language_encoder_cfg,
                 load_path=None):
        super().__init__()
        self.dim = dim = transformer_cfg.dim
        self.transformer = self._init_transformer(**transformer_cfg)
        self.track_proj_encoder, self.track_decoder = self._init_track_modules(**track_cfg, dim=dim)
        self.img_proj_encoder, self.img_decoder = self._init_video_modules(**vid_cfg, dim=dim)
        self.language_encoder = self._init_language_encoder(output_size=dim, **language_encoder_cfg)
        self._init_weights(self.dim, self.num_img_patches)

        if load_path is not None:
            self.load(load_path)
            print(f"loaded model from {load_path}")

    def _init_transformer(self, dim, dim_head, heads, depth, attn_dropout, ff_dropout):
        self.transformer = Transformer(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            depth=depth,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout)

        return self.transformer

    def _init_track_modules(self, dim, num_track_ts, num_track_ids, patch_size=1):
        self.num_track_ts = num_track_ts
        self.num_track_ids = num_track_ids
        self.track_patch_size = patch_size

        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=num_track_ts,
            num_track_ids=num_track_ids,
            patch_size=patch_size,
            in_dim=2,
            embed_dim=dim)
        self.num_track_patches = self.track_proj_encoder.num_patches
        self.track_decoder = nn.Linear(dim, 2 * patch_size, bias=True)
        self.num_track_ids = num_track_ids
        self.num_track_ts = num_track_ts

        return self.track_proj_encoder, self.track_decoder

    def _init_video_modules(self, dim, img_size, patch_size, frame_stack=1, img_mean=[.5, .5, .5], img_std=[.5, .5, .5]):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        else:
            img_size = (img_size[0], img_size[1])
        self.img_size = img_size
        self.frame_stack = frame_stack
        self.patch_size = patch_size
        self.img_proj_encoder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=3 * self.frame_stack,
            embed_dim=dim,
        )
        self.num_img_patches = self.img_proj_encoder.num_patches
        self.img_decoder = nn.Linear(dim, 3 * self.frame_stack * patch_size ** 2, bias=True)

        return self.img_proj_encoder, self.img_decoder

    def _init_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _init_weights(self, dim, num_img_patches):
        """
        initialize weights; freeze all positional embeddings
        """
        num_track_t = self.num_track_ts // self.track_patch_size

        self.track_embed = nn.Parameter(torch.randn(1, num_track_t, 1, dim), requires_grad=True)
        self.img_embed = nn.Parameter(torch.randn(1, num_img_patches, dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.randn(1, 1, dim))

        track_embed = get_1d_sincos_pos_embed(dim, num_track_t)
        track_embed = rearrange(track_embed, 't d -> () t () d')
        self.track_embed.data.copy_(torch.from_numpy(track_embed))

        num_patches_h, num_patches_w = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        img_embed = get_2d_sincos_pos_embed(dim, (num_patches_h, num_patches_w))
        img_embed = rearrange(img_embed, 'n d -> () n d')
        self.img_embed.data.copy_(torch.from_numpy(img_embed))

        print(f"num_track_patches: {self.num_track_patches}, num_img_patches: {num_img_patches}, total: {self.num_track_patches + num_img_patches}")

    def _preprocess_track(self, track):
        return track

    def _preprocess_vis(self, vis):
        return vis

    def _preprocess_vid(self, vid):
        assert torch.max(vid) >= 2

        vid = vid[:, -self.frame_stack:]
        vid = self.img_normalizer(vid / 255.)
        return vid

    def _encode_track(self, track):
        """
        track: (b, t, n, 2)
        """
        b, t, n, _ = track.shape
        track = self._mask_track_as_first(track)  # b, t, n, d. track embedding is 1, t, 1, d
        track = self.track_proj_encoder(track)

        track = track + self.track_embed
        track = rearrange(track, 'b t n d -> b (t n) d')
        return track

    def _encode_video(self, vid, p):
        """
        vid: (b, t, c, h, w)
        """
        vid = rearrange(vid, "b t c h w -> b (t c) h w")
        patches = self.img_proj_encoder(vid)  # b, n, d
        patches = self._mask_patches(patches, p=p)
        patches = patches + self.img_embed

        return patches

    def _mask_patches(self, patches, p):
        """
        mask patches according to p
        """
        b, n, _ = patches.shape
        mask = torch.rand(b, n, device=patches.device) < p
        masked_patches = patches.clone()
        masked_patches[mask] = self.mask_token
        return masked_patches

    def _mask_track_as_first(self, track):
        """
        mask out all frames to have the same token as the first frame
        """
        mask_track = track.clone() # b, t, n, d
        mask_track[:, 1:] = track[:, [0]]
        return mask_track

    def forward(self, vid, track, task_emb, p_img):
        """
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb, (b, emb_size)
        """
        assert torch.max(vid) <=1.
        B, T, _, _ = track.shape
        patches = self._encode_video(vid, p_img)  # (b, n_image, d)
        enc_track = self._encode_track(track)

        text_encoded = self.language_encoder(task_emb)  # (b, c)
        text_encoded = rearrange(text_encoded, 'b c -> b 1 c')

        x = torch.cat([enc_track, patches, text_encoded], dim=1)
        x = self.transformer(x)

        rec_track, rec_patches = x[:, :self.num_track_patches], x[:, self.num_track_patches:-1]
        rec_patches = self.img_decoder(rec_patches)  # (b, n_image, 3 * t * patch_size ** 2)
        rec_track = self.track_decoder(rec_track)  # (b, (t n), 2 * patch_size)
        num_track_h = self.num_track_ts // self.track_patch_size
        rec_track = rearrange(rec_track, 'b (t n) (p c) -> b (t p) n c', p=self.track_patch_size, t=num_track_h)

        return rec_track, rec_patches

    def reconstruct(self, vid, track, task_emb, p_img):
        """
        wrapper of forward with preprocessing
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb: (b, e)
        """
        assert len(vid.shape) == 5  # b, t, c, h, w
        track = self._preprocess_track(track)
        vid = self._preprocess_vid(vid)
        return self.forward(vid, track, task_emb, p_img)

    def forward_loss(self,
                     vid,
                     track,
                     task_emb,
                     lbd_track,
                     lbd_img,
                     p_img,
                     return_outs=False,
                     vis=None):
        """
        track: (b, tl, n, 2), which means current time step t0 -> t0 + tl
        vid: (b, t, c, h, w), which means the past time step t0 - t -> t0
        task_emb: (b, e)
        """

        b, tl, n, _ = track.shape
        if vis is None:
            vis = torch.ones((b, tl, n)).to(track.device)

        track = self._preprocess_track(track)
        vid = self._preprocess_vid(vid)
        vis = self._preprocess_vis(vis)

        rec_track, rec_patches = self.forward(vid, track, task_emb, p_img)
        vis[vis == 0] = .1
        vis = repeat(vis, "b tl n -> b tl n c", c=2)

        track_loss = torch.mean((rec_track - track) ** 2 * vis)
        img_loss = torch.mean((rec_patches - self._patchify(vid)) ** 2)
        loss = lbd_track * track_loss + lbd_img * img_loss

        ret_dict = {
            "loss": loss.item(),
            "track_loss": track_loss.item(),
            "img_loss": img_loss.item(),
        }

        if return_outs:
            return loss.sum(), ret_dict, (rec_track, rec_patches)
        return loss.sum(), ret_dict

    def forward_vis(self, vid, track, task_emb, p_img):
        """
        track: (b, tl, n, 2)
        vid: (b, t, c, h, w)
        """
        b = vid.shape[0]
        assert b == 1, "only support batch size 1 for visualization"

        H, W = self.img_size
        _vid = vid.clone()
        track = self._preprocess_track(track)
        vid = self._preprocess_vid(vid)

        rec_track, rec_patches = self.forward(vid, track, task_emb, p_img)
        track_loss = F.mse_loss(rec_track, track)
        img_loss = F.mse_loss(rec_patches, self._patchify(vid))
        loss = track_loss + img_loss

        rec_image = self._unpatchify(rec_patches)

        # place them side by side
        combined_image = torch.cat([vid[:, -1], rec_image[:, -1]], dim=-1)  # only visualize the current frame
        combined_image = self.img_unnormalizer(combined_image) * 255
        combined_image = torch.clamp(combined_image, 0, 255)
        combined_image = rearrange(combined_image, '1 c h w -> h w c')

        track = track.clone()
        rec_track = rec_track.clone()

        rec_track_vid = tracks_to_video(rec_track, img_size=H)
        track_vid = tracks_to_video(track, img_size=H)

        combined_track_vid = torch.cat([track_vid, rec_track_vid], dim=-1)

        _vid = torch.cat([_vid, _vid], dim=-1)
        combined_track_vid = _vid * .25 + combined_track_vid * .75

        ret_dict = {
            "loss": loss.sum().item(),
            "track_loss": track_loss.sum().item(),
            "img_loss": img_loss.sum().item(),
            "combined_image": combined_image.cpu().numpy().astype(np.uint8),
            "combined_track_vid": combined_track_vid.cpu().numpy().astype(np.uint8),
        }

        return loss.sum(), ret_dict

    def _patchify(self, imgs):
        """
        imgs: (N, T, 3, H, W)
        x: (N, L, patch_size**2 * T * 3)
        """
        N, T, C, img_H, img_W = imgs.shape
        p = self.img_proj_encoder.patch_size[0]
        assert img_H % p == 0 and img_W % p == 0

        h = img_H // p
        w = img_W // p
        x = imgs.reshape(shape=(imgs.shape[0], T, C, h, p, w, p))
        x = rearrange(x, "n t c h p w q -> n h w p q t c")
        x = rearrange(x, "n h w p q t c -> n (h w) (p q t c)")
        return x

    def _unpatchify(self, x):
        """
        x: (N, L, patch_size**2 * T * 3)
        imgs: (N, T, 3, H, W)
        """
        p = self.img_proj_encoder.patch_size[0]
        h = self.img_size[0] // p
        w = self.img_size[1] // p
        assert h * w == x.shape[1]

        x = rearrange(x, "n (h w) (p q t c) -> n h w p q t c", h=h, w=w, p=p, q=p, t=self.frame_stack, c=3)
        x = rearrange(x, "n h w p q t c -> n t c h p w q")
        imgs = rearrange(x, "n t c h p w q -> n t c (h p) (w q)")
        return imgs

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))
