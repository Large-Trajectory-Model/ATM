import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torchvision.transforms as T

from einops import rearrange, repeat

from atm.model import *
from atm.model.track_patch_embed import TrackPatchEmbed
from atm.policy.vilt_modules.transformer_modules import *
from atm.policy.vilt_modules.rgb_modules import *
from atm.policy.vilt_modules.language_modules import *
from atm.policy.vilt_modules.extra_state_modules import ExtraModalityTokens
from atm.policy.vilt_modules.policy_head import *
from atm.utils.flow_utils import ImageUnNormalize, sample_double_grid, tracks_to_video

###############################################################################
#
# A ViLT Policy
#
###############################################################################


class BCViLTPolicy(nn.Module):
    """
    Input: (o_{t-H}, ... , o_t)
    Output: a_t or distribution of a_t
    """

    def __init__(self, obs_cfg, img_encoder_cfg, language_encoder_cfg, extra_state_encoder_cfg, track_cfg,
                 spatial_transformer_cfg, temporal_transformer_cfg,
                 policy_head_cfg, load_path=None):
        super().__init__()

        self._process_obs_shapes(**obs_cfg)

        # 1. encode image
        self._setup_image_encoder(**img_encoder_cfg)

        # 2. encode language (spatial)
        self.language_encoder_spatial = self._setup_language_encoder(output_size=self.spatial_embed_size, **language_encoder_cfg)

        # 3. Track Transformer module
        self._setup_track(**track_cfg)

        # 3. define spatial positional embeddings, modality embeddings, and spatial token for summary
        self._setup_spatial_positional_embeddings()

        # 4. define spatial transformer
        self._setup_spatial_transformer(**spatial_transformer_cfg)

        ### 5. encode extra information (e.g. gripper, joint_state)
        self.extra_encoder = self._setup_extra_state_encoder(extra_embedding_size=self.temporal_embed_size, **extra_state_encoder_cfg)

        # 6. encode language (temporal), this will also act as the TEMPORAL_TOKEN, i.e., CLS token for action prediction
        self.language_encoder_temporal = self._setup_language_encoder(output_size=self.temporal_embed_size, **language_encoder_cfg)

        # 7. define temporal transformer
        self._setup_temporal_transformer(**temporal_transformer_cfg)

        # 8. define policy head
        self._setup_policy_head(**policy_head_cfg)

        if load_path is not None:
            self.load(load_path)
            self.track.load(f"{track_cfg.track_fn}/model_best.ckpt")

    def _process_obs_shapes(self, obs_shapes, num_views, extra_states, img_mean, img_std, max_seq_len):
        self.img_normalizer = T.Normalize(img_mean, img_std)
        self.img_unnormalizer = ImageUnNormalize(img_mean, img_std)
        self.obs_shapes = obs_shapes
        self.policy_num_track_ts = obs_shapes["tracks"][0]
        self.policy_num_track_ids = obs_shapes["tracks"][1]
        self.num_views = num_views
        self.extra_state_keys = extra_states
        self.max_seq_len = max_seq_len
        # define buffer queue for encoded latent features
        self.latent_queue = deque(maxlen=max_seq_len)
        self.track_obs_queue = deque(maxlen=max_seq_len)

    def _setup_image_encoder(self, network_name, patch_size, embed_size, no_patch_embed_bias):
        self.spatial_embed_size = embed_size
        self.image_encoders = []
        for _ in range(self.num_views):
            input_shape = self.obs_shapes["rgb"]
            self.image_encoders.append(eval(network_name)(input_shape=input_shape, patch_size=patch_size,
                                                          embed_size=self.spatial_embed_size,
                                                          no_patch_embed_bias=no_patch_embed_bias))
        self.image_encoders = nn.ModuleList(self.image_encoders)

        self.img_num_patches = sum([x.num_patches for x in self.image_encoders])

    def _setup_language_encoder(self, network_name, **language_encoder_kwargs):
        return eval(network_name)(**language_encoder_kwargs)

    def _setup_track(self, track_fn, policy_track_patch_size=None, use_zero_track=False):
        """
        track_fn: path to the track model
        policy_track_patch_size: The patch size of TrackPatchEmbedding in the policy, if None, it will be assigned the same patch size as TrackTransformer by default
        use_zero_track: whether to zero out the tracks (ie use only the image)
        """
        track_cfg = OmegaConf.load(f"{track_fn}/config.yaml")
        self.use_zero_track = use_zero_track

        track_cfg.model_cfg.load_path = f"{track_fn}/model_best.ckpt"
        track_cls = eval(track_cfg.model_name)
        self.track = track_cls(**track_cfg.model_cfg)
        # freeze
        self.track.eval()
        for param in self.track.parameters():
            param.requires_grad = False

        self.num_track_ids = self.track.num_track_ids
        self.num_track_ts = self.track.num_track_ts
        self.policy_track_patch_size = self.track.track_patch_size if policy_track_patch_size is None else policy_track_patch_size


        self.track_proj_encoder = TrackPatchEmbed(
            num_track_ts=self.policy_num_track_ts,
            num_track_ids=self.num_track_ids,
            patch_size=self.policy_track_patch_size,
            in_dim=2 + self.num_views,  # X, Y, one-hot view embedding
            embed_dim=self.spatial_embed_size)

        self.track_id_embed_dim = 16
        self.num_track_patches_per_view = self.track_proj_encoder.num_patches_per_track
        self.num_track_patches = self.num_track_patches_per_view * self.num_views

    def _setup_spatial_positional_embeddings(self):
        # setup positional embeddings
        spatial_token = nn.Parameter(torch.randn(1, 1, self.spatial_embed_size))  # SPATIAL_TOKEN
        img_patch_pos_embed = nn.Parameter(torch.randn(1, self.img_num_patches, self.spatial_embed_size))
        track_patch_pos_embed = nn.Parameter(torch.randn(1, self.num_track_patches, self.spatial_embed_size-self.track_id_embed_dim))
        modality_embed = nn.Parameter(
            torch.randn(1, len(self.image_encoders) + self.num_views + 1, self.spatial_embed_size)
        )  # IMG_PATCH_TOKENS + TRACK_PATCH_TOKENS + SENTENCE_TOKEN

        self.register_parameter("spatial_token", spatial_token)
        self.register_parameter("img_patch_pos_embed", img_patch_pos_embed)
        self.register_parameter("track_patch_pos_embed", track_patch_pos_embed)
        self.register_parameter("modality_embed", modality_embed)

        # for selecting modality embed
        modality_idx = []
        for i, encoder in enumerate(self.image_encoders):
            modality_idx += [i] * encoder.num_patches
        for i in range(self.num_views):
            modality_idx += [modality_idx[-1] + 1] * self.num_track_ids * self.num_track_patches_per_view  # for track embedding
        modality_idx += [modality_idx[-1] + 1]  # for sentence embedding
        self.modality_idx = torch.LongTensor(modality_idx)

    def _setup_extra_state_encoder(self, **extra_state_encoder_cfg):
        if len(self.extra_state_keys) == 0:
            return None
        else:
            return ExtraModalityTokens(
                use_joint=("joint_states" in self.extra_state_keys),
                use_gripper=("gripper_states" in self.extra_state_keys),
                use_ee=("ee_states" in self.extra_state_keys),
                **extra_state_encoder_cfg
            )

    def _setup_spatial_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout,
                                   spatial_downsample, spatial_downsample_embed_size, use_language_token=True):
        self.spatial_transformer = TransformerDecoder(
            input_size=self.spatial_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,
        )

        if spatial_downsample:
            self.temporal_embed_size = spatial_downsample_embed_size
            self.spatial_downsample = nn.Linear(self.spatial_embed_size, self.temporal_embed_size)
        else:
            self.temporal_embed_size = self.spatial_embed_size
            self.spatial_downsample = nn.Identity()

        self.spatial_transformer_use_text = use_language_token

    def _setup_temporal_transformer(self, num_layers, num_heads, head_output_size, mlp_hidden_size, dropout, use_language_token=True):
        self.temporal_position_encoding_fn = SinusoidalPositionEncoding(input_size=self.temporal_embed_size)

        self.temporal_transformer = TransformerDecoder(
            input_size=self.temporal_embed_size,
            num_layers=num_layers,
            num_heads=num_heads,
            head_output_size=head_output_size,
            mlp_hidden_size=mlp_hidden_size,
            dropout=dropout,)
        self.temporal_transformer_use_text = use_language_token

        action_cls_token = nn.Parameter(torch.zeros(1, 1, self.temporal_embed_size))
        nn.init.normal_(action_cls_token, std=1e-6)
        self.register_parameter("action_cls_token", action_cls_token)

    def _setup_policy_head(self, network_name, **policy_head_kwargs):
        policy_head_kwargs["input_size"] \
            = self.temporal_embed_size + self.num_views * self.policy_num_track_ts * self.policy_num_track_ids * 2

        action_shape = policy_head_kwargs["output_size"]
        self.act_shape = action_shape
        self.out_shape = np.prod(action_shape)
        policy_head_kwargs["output_size"] = self.out_shape
        self.policy_head = eval(network_name)(**policy_head_kwargs)

    @torch.no_grad()
    def preprocess(self, obs, track, action):
        """
        Preprocess observations, according to an observation dictionary.
        Return the feature and state.
        """
        b, v, t, c, h, w = obs.shape

        action = action.reshape(b, t, self.out_shape)

        obs = self._preprocess_rgb(obs)

        return obs, track, action

    @torch.no_grad()
    def _preprocess_rgb(self, rgb):
        rgb = self.img_normalizer(rgb / 255.)
        return rgb

    def _get_view_one_hot(self, tr):
        """ tr: b, v, t, tl, n, d -> (b, v, t), tl n, d + v"""
        b, v, t, tl, n, d = tr.shape
        tr = rearrange(tr, "b v t tl n d -> (b t tl n) v d")
        one_hot = torch.eye(v, device=tr.device, dtype=tr.dtype)[None, :, :].repeat(tr.shape[0], 1, 1)
        tr_view = torch.cat([tr, one_hot], dim=-1)  # (b t tl n) v (d + v)
        tr_view = rearrange(tr_view, "(b t tl n) v c -> b v t tl n c", b=b, v=v, t=t, tl=tl, n=n, c=d + v)
        return tr_view

    def track_encode(self, track_obs, task_emb):
        """
        Args:
            track_obs: b v t tt_fs c h w
            task_emb: b e
        Returns: b v t track_len n 2
        """
        assert self.num_track_ids == 32
        b, v, t, *_ = track_obs.shape

        if self.use_zero_track:
            recon_tr = torch.zeros((b, v, t, self.num_track_ts, self.num_track_ids, 2), device=track_obs.device, dtype=track_obs.dtype)
        else:
            track_obs_to_pred = rearrange(track_obs, "b v t fs c h w -> (b v t) fs c h w")

            grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
            grid_sampled_track = repeat(grid_points, "n d -> b v t tl n d", b=b, v=v, t=t, tl=self.num_track_ts)
            grid_sampled_track = rearrange(grid_sampled_track, "b v t tl n d -> (b v t) tl n d")

            expand_task_emb = repeat(task_emb, "b e -> b v t e", b=b, v=v, t=t)
            expand_task_emb = rearrange(expand_task_emb, "b v t e -> (b v t) e")
            with torch.no_grad():
                pred_tr, _ = self.track.reconstruct(track_obs_to_pred, grid_sampled_track, expand_task_emb, p_img=0)  # (b v t) tl n d
                recon_tr = rearrange(pred_tr, "(b v t) tl n d -> b v t tl n d", b=b, v=v, t=t)

        recon_tr = recon_tr[:, :, :, :self.policy_num_track_ts, :, :]  # truncate the track to a shorter one
        _recon_tr = recon_tr.clone()  # b v t tl n 2
        with torch.no_grad():
            tr_view = self._get_view_one_hot(recon_tr)  # b v t tl n c

        tr_view = rearrange(tr_view, "b v t tl n c -> (b v t) tl n c")
        tr = self.track_proj_encoder(tr_view)  # (b v t) track_patch_num n d
        tr = rearrange(tr, "(b v t) pn n d -> (b t n) (v pn) d", b=b, v=v, t=t, n=self.num_track_ids)  # (b t n) (v patch_num) d

        return tr, _recon_tr

    def spatial_encode(self, obs, track_obs, task_emb, extra_states, return_recon=False):
        """
        Encode the images separately in the videos along the spatial axis.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w, (0, 255)
            task_emb: b e
            extra_states: {k: b t n}
        Returns: out: (b t 2+num_extra c), recon_track: (b v t tl n 2)
        """
        # 1. encode image
        img_encoded = []
        for view_idx in range(self.num_views):
            img_encoded.append(
                rearrange(
                    TensorUtils.time_distributed(
                        obs[:, view_idx, ...], self.image_encoders[view_idx]
                    ),
                    "b t c h w -> b t (h w) c",
                )
            )  # (b, t, num_patches, c)

        img_encoded = torch.cat(img_encoded, -2)  # (b, t, 2*num_patches, c)
        img_encoded += self.img_patch_pos_embed.unsqueeze(0)  # (b, t, 2*num_patches, c)
        B, T = img_encoded.shape[:2]

        # 2. encode task_emb
        text_encoded = self.language_encoder_spatial(task_emb)  # (b, c)
        text_encoded = text_encoded.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c)

        # 3. encode track
        track_encoded, _recon_track = self.track_encode(track_obs, task_emb)  # track_encoded: ((b t n), 2*patch_num, c)  _recon_track: (b, v, track_len, n, 2)
        # patch position embedding
        tr_feat, tr_id_emb = track_encoded[:, :, :-self.track_id_embed_dim], track_encoded[:, :, -self.track_id_embed_dim:]
        tr_feat += self.track_patch_pos_embed  # ((b t n), 2*patch_num, c)
        # track id embedding
        tr_id_emb[:, 1:, -self.track_id_embed_dim:] = tr_id_emb[:, :1, -self.track_id_embed_dim:]  # guarantee the permutation invariance
        track_encoded = torch.cat([tr_feat, tr_id_emb], dim=-1)
        track_encoded = rearrange(track_encoded, "(b t n) pn d -> b t (n pn) d", b=B, t=T)  # (b, t, 2*num_track*num_track_patch, c)

        # 3. concat img + track + text embs then add modality embeddings
        if self.spatial_transformer_use_text:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded, text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 1, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx, :]
        else:
            img_track_text_encoded = torch.cat([img_encoded, track_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch, c)
            img_track_text_encoded += self.modality_embed[None, :, self.modality_idx[:-1], :]

        # 4. add spatial token
        spatial_token = self.spatial_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c)
        encoded = torch.cat([spatial_token, img_track_text_encoded], -2)  # (b, t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)

        # 5. pass through transformer
        encoded = rearrange(encoded, "b t n c -> (b t) n c")  # (b*t, 2*num_img_patch + 2*num_track*num_track_patch + 2, c)
        out = self.spatial_transformer(encoded)
        out = out[:, 0]  # extract spatial token as summary at o_t
        out = self.spatial_downsample(out).view(B, T, 1, -1)  # (b, t, 1, c')

        # 6. encode extra states
        if self.extra_encoder is None:
            extra = None
        else:
            extra = self.extra_encoder(extra_states)  # (B, T, num_extra, c')

        # 7. encode language, treat it as action token
        text_encoded_ = self.language_encoder_temporal(task_emb)  # (b, c')
        text_encoded_ = text_encoded_.view(B, 1, 1, -1).expand(-1, T, -1, -1)  # (b, t, 1, c')
        action_cls_token = self.action_cls_token.unsqueeze(0).expand(B, T, -1, -1)  # (b, t, 1, c')
        if self.temporal_transformer_use_text:
            out_seq = [action_cls_token, text_encoded_, out]
        else:
            out_seq = [action_cls_token, out]

        if self.extra_encoder is not None:
            out_seq.append(extra)
        output = torch.cat(out_seq, -2)  # (b, t, 2 or 3 + num_extra, c')

        if return_recon:
            output = (output, _recon_track)

        return output

    def temporal_encode(self, x):
        """
        Args:
            x: b, t, num_modality, c
        Returns:
        """
        pos_emb = self.temporal_position_encoding_fn(x)  # (t, c)
        x = x + pos_emb.unsqueeze(1)  # (b, t, 2+num_extra, c)
        sh = x.shape
        self.temporal_transformer.compute_mask(x.shape)

        x = TensorUtils.join_dimensions(x, 1, 2)  # (b, t*num_modality, c)
        x = self.temporal_transformer(x)
        x = x.reshape(*sh)  # (b, t, num_modality, c)
        return x[:, :, 0]  # (b, t, c)

    def forward(self, obs, track_obs, track, task_emb, extra_states):
        """
        Return feature and info.
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            extra_states: {k: b t e}
        """
        x, recon_track = self.spatial_encode(obs, track_obs, task_emb, extra_states, return_recon=True)  # x: (b, t, 2+num_extra, c), recon_track: (b, v, t, tl, n, 2)
        x = self.temporal_encode(x)  # (b, t, c)

        recon_track = rearrange(recon_track, "b v t tl n d -> b t (v tl n d)")
        x = torch.cat([x, recon_track], dim=-1)  # (b, t, c + v*tl*n*2)

        dist = self.policy_head(x)  # only use the current timestep feature to predict action
        return dist

    def forward_loss(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2, not used for training, only preserved for unified interface
            task_emb: b emb_size
            action: b t act_dim
        """
        obs, track, action = self.preprocess(obs, track, action)
        dist = self.forward(obs, track_obs, track, task_emb, extra_states)
        loss = self.policy_head.loss_fn(dist, action, reduction="mean")

        ret_dict = {
            "bc_loss": loss.sum().item(),
        }

        if not self.policy_head.deterministic:
            # pseudo loss
            sampled_action = dist.sample().detach()
            mse_loss = F.mse_loss(sampled_action, action)
            ret_dict["pseudo_sampled_action_mse_loss"] = mse_loss.sum().item()

        ret_dict["loss"] = ret_dict["bc_loss"]
        return loss.sum(), ret_dict

    def forward_vis(self, obs, track_obs, track, task_emb, extra_states, action):
        """
        Args:
            obs: b v t c h w
            track_obs: b v t tt_fs c h w
            track: b v t track_len n 2
            task_emb: b emb_size
        Returns:
        """
        _, track, _ = self.preprocess(obs, track, action)
        track = track[:, :, 0, :, :, :]  # (b, v, track_len, n, 2) use the track in the first timestep

        b, v, t, track_obs_t, c, h, w = track_obs.shape
        if t >= self.num_track_ts:
            track_obs = track_obs[:, :, :self.num_track_ts, ...]
            track = track[:, :, :self.num_track_ts, ...]
        else:
            last_obs = track_obs[:, :, -1:, ...]
            pad_obs = repeat(last_obs, "b v 1 track_obs_t c h w -> b v t track_obs_t c h w", t=self.num_track_ts-t)
            track_obs = torch.cat([track_obs, pad_obs], dim=2)
            last_track = track[:, :, -1:, ...]
            pad_track = repeat(last_track, "b v 1 n d -> b v tl n d", tl=self.num_track_ts-t)
            track = torch.cat([track, pad_track], dim=2)

        grid_points = sample_double_grid(4, device=track_obs.device, dtype=track_obs.dtype)
        grid_track = repeat(grid_points, "n d -> b v tl n d", b=b, v=v, tl=self.num_track_ts)

        all_ret_dict = {}
        for view in range(self.num_views):
            gt_track = track[:1, view]  # (1 tl n d)
            gt_track_vid = tracks_to_video(gt_track, img_size=h)
            combined_gt_track_vid = (track_obs[:1, view, 0, :, ...] * .25 + gt_track_vid * .75).cpu().numpy().astype(np.uint8)

            _, ret_dict = self.track.forward_vis(track_obs[:1, view, 0, :, ...], grid_track[:1, view], task_emb[:1], p_img=0)
            ret_dict["combined_track_vid"] = np.concatenate([combined_gt_track_vid, ret_dict["combined_track_vid"]], axis=-1)

            all_ret_dict = {k: all_ret_dict.get(k, []) + [v] for k, v in ret_dict.items()}

        for k, v in all_ret_dict.items():
            if k == "combined_image" or k == "combined_track_vid":
                all_ret_dict[k] = np.concatenate(v, axis=-2)  # concat on the height dimension
            else:
                all_ret_dict[k] = np.mean(v)
        return None, all_ret_dict

    def act(self, obs, task_emb, extra_states):
        """
        Args:
            obs: (b, v, h, w, c)
            task_emb: (b, em_dim)
            extra_states: {k: (b, state_dim,)}
        """
        self.eval()
        B = obs.shape[0]

        # expand time dimenstion
        obs = rearrange(obs, "b v h w c -> b v 1 c h w").copy()
        extra_states = {k: rearrange(v, "b e -> b 1 e") for k, v in extra_states.items()}

        dtype = next(self.parameters()).dtype
        device = next(self.parameters()).device
        obs = torch.Tensor(obs).to(device=device, dtype=dtype)
        task_emb = torch.Tensor(task_emb).to(device=device, dtype=dtype)
        extra_states = {k: torch.Tensor(v).to(device=device, dtype=dtype) for k, v in extra_states.items()}

        if (obs.shape[-2] != self.obs_shapes["rgb"][-2]) or (obs.shape[-1] != self.obs_shapes["rgb"][-1]):
            obs = rearrange(obs, "b v fs c h w -> (b v fs) c h w")
            obs = F.interpolate(obs, size=self.obs_shapes["rgb"][-2:], mode="bilinear", align_corners=False)
            obs = rearrange(obs, "(b v fs) c h w -> b v fs c h w", b=B, v=self.num_views)

        while len(self.track_obs_queue) < self.max_seq_len:
            self.track_obs_queue.append(torch.zeros_like(obs))
        self.track_obs_queue.append(obs.clone())
        track_obs = torch.cat(list(self.track_obs_queue), dim=2)  # b v fs c h w
        track_obs = rearrange(track_obs, "b v fs c h w -> b v 1 fs c h w")

        obs = self._preprocess_rgb(obs)

        with torch.no_grad():
            x, rec_tracks = self.spatial_encode(obs, track_obs, task_emb=task_emb, extra_states=extra_states, return_recon=True)  # x: (b, 1, 4, c), recon_track: (b, v, 1, tl, n, 2)
            self.latent_queue.append(x)
            x = torch.cat(list(self.latent_queue), dim=1)  # (b, t, 4, c)
            x = self.temporal_encode(x)  # (b, t, c)

            feat = torch.cat([x[:, -1], rearrange(rec_tracks[:, :, -1, :, :, :], "b v tl n d -> b (v tl n d)")], dim=-1)

            action = self.policy_head.get_action(feat)  # only use the current timestep feature to predict action
            action = action.detach().cpu()  # (b, act_dim)

        action = action.reshape(-1, *self.act_shape)
        action = torch.clamp(action, -1, 1)
        return action.float().cpu().numpy(), (None, rec_tracks[:, :, -1, :, :, :])  # (b, *act_shape)

    def reset(self):
        self.latent_queue.clear()
        self.track_obs_queue.clear()

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path, map_location="cpu"))

    def train(self, mode=True):
        super().train(mode)
        self.track.eval()

    def eval(self):
        super().eval()
        self.track.eval()
