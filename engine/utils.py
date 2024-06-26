import os
from typing  import List
import torch
import numpy as np
from tqdm import tqdm
import wandb
from PIL import Image
from einops import rearrange
from atm.utils.flow_utils import combine_track_and_img, draw_traj_on_images
from atm.utils.video_utils import video_pad_time


obs_key_mapping = {
    "gripper_states": "robot0_gripper_qpos",
    "joint_states": "robot0_joint_pos",
}


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frame, success, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (b, c, h, w)
        success: (b, 1)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    if any(success):
        b, c, h, w = frame.shape
        color = np.array(color, dtype=frame.dtype)[None, :, None, None]
        boundary = int(min(h, w) * 0.015)
        frame[success, :, :boundary, :] = color
        frame[success, :, -boundary:, :] = color
        frame[success, :, :, :boundary] = color
        frame[success, :, :, -boundary:] = color
    return frame


@torch.no_grad()
def rollout(env_dict, policy, num_env_rollouts, horizon=None, return_wandb_video=True,
            success_vid_first=False, fail_vid_first=False, connect_points_with_line=False):
    policy.eval()
    all_env_indices = []
    all_env_rewards = []
    all_env_succ = []
    all_env_horizon = []
    env_vid = []
    env_additional_metrics = []
    all_env_descriptions = []

    for env_description, (env_idx, env) in env_dict.items():
        all_env_indices.append(env_idx)
        all_rewards = []
        all_succ = []
        all_horizon = []
        vid = []
        additional_metrics = {}
        for _ in tqdm(range(num_env_rollouts)):
            reward = None
            success = False
            last_info = None
            episode_frames = []
            obs = env.reset()
            policy.reset()
            done = False
            step_i = 0
            while not done and (horizon is None or step_i < horizon):
                rgb = obs["image"]  # (b, v, h, w, c)
                task_emb = obs.get("task_emb", None)
                extra_states = {k: obs[obs_key_mapping[k]] for k in policy.extra_state_keys}
                a, _tracks = policy.act(rgb, task_emb, extra_states)
                obs, r, done, info = env.step(a)
                reward = list(r) if reward is None else [old_r + new_r for old_r, new_r in zip(reward, r)]
                done = all(done)
                success = list(info["success"])

                video_img = rearrange(rgb.copy(), "b v h w c -> b v c h w")
                b, _, c, h, w = video_img.shape

                if _tracks is not None:
                    _track, _rec_track = _tracks
                    if connect_points_with_line:
                        base_track_img = draw_traj_on_images(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = draw_traj_on_images(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8)*255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                    else:
                        base_track_img = combine_track_and_img(_rec_track[:, 0], video_img[:, 0])  # (b, c, h, w)
                        wrist_track_img = combine_track_and_img(_rec_track[:, 1], video_img[:, 1])
                        frame = np.concatenate([base_track_img, np.ones((b, c, h, 2), dtype=np.uint8) * 255, wrist_track_img], axis=-1)  # (b, c, h, 2w)
                else:
                    frame = np.concatenate([video_img[:, 0], np.ones((b, c, h, w), dtype=np.uint8)*255, video_img[:, 1]], axis=-1)  # (b, c, h, 2w)

                frame = render_done_to_boundary(frame, success)
                episode_frames.append(frame)

                step_i += 1

                last_info = info
                if done or (horizon is not None and step_i >= horizon):
                    break

            episode_videos = np.stack(episode_frames, axis=1)  # (b, t, c, h, w)
            vid.extend(list(episode_videos))  # b*[(t, c, h, w)]

            all_rewards += reward
            all_horizon += [step_i + 1]
            all_succ += success

        if len(additional_metrics) == 0:
            additional_metrics = {k: [v] for k, v in last_info.items() if k != "success"}
        else:
            for k, v in additional_metrics.items():
                additional_metrics[k].append(last_info[k])

        vid = video_pad_time(vid)  # (b, t, c, h, w)
        vid, rearrange_idx = rearrange_videos(vid, all_succ, success_vid_first, fail_vid_first)
        all_rewards = np.array(all_rewards)[rearrange_idx].astype(np.float32)
        all_succ = np.array(all_succ)[rearrange_idx].astype(np.float32)

        all_env_rewards.append(all_rewards)
        all_env_succ.append(all_succ)
        all_env_horizon.append(all_horizon)
        env_vid.append(video_pad_time(vid))  # [(b, t, c, h, w)]
        env_additional_metrics.append(additional_metrics)
        all_env_descriptions.append(env_description)

    results = {}
    for idx, env_idx in enumerate(all_env_indices):
        results[f"rollout/return_env{env_idx}"] = np.mean(all_env_rewards[idx])
        results[f"rollout/horizon_env{env_idx}"] = np.mean(all_env_horizon[idx])
        results[f"rollout/success_env{env_idx}"] = np.mean(all_env_succ[idx])
        if return_wandb_video:
            results[f"rollout/vis_env{env_idx}"] = wandb.Video(env_vid[idx], fps=30, format="mp4", caption=all_env_descriptions[idx])
        else:
            results[f"rollout/vis_env{env_idx}"] = env_vid[idx]
        for k, v in env_additional_metrics[idx].items():
            results[f"rollout/{k}_env{env_idx}"] = np.mean(v)

    return results


def merge_results(results: List[dict], compute_avg=True):
    merged_results = {}
    for result_dict in results:
        for k, v in result_dict.items():
            if k in merged_results:
                if isinstance(v, list):
                    merged_results[k].append(v)
                else:
                    merged_results[k] = [merged_results[k], v]
            else:
                merged_results[k] = v

    if compute_avg:
        merged_results["rollout/return_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/return_env" in k]).flatten())
        merged_results["rollout/horizon_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/horizon_env" in k]).flatten())
        merged_results["rollout/success_env_avg"] = np.mean(np.array([v for k, v in merged_results.items() if "rollout/success_env" in k]).flatten())
    return merged_results
