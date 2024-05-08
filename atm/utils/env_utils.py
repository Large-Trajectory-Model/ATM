import json
import copy
import math
from collections import OrderedDict

import numpy as np
from collections.abc import Iterable
from libero.utils.env_utils import make_libero_env
from robosuite.wrappers import Wrapper


class ObservationWrapper(Wrapper):

    valid_obs_types = ["image"]

    def __init__(self, env, masks, cameras):
        super(ObservationWrapper, self).__init__(env)
        self.masks = masks
        self.cameras = cameras

    def reset(self):
        obs = self.env.reset()
        obs_dict = self._stack_obs(obs)
        return obs_dict

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_dict = self._stack_obs(obs)
        return obs_dict, reward, done, info

    def _stack_obs(self, obs):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=0)
        return obs_dict


class LiberoImageUpsideDownWrapper(Wrapper):
    def __init__(self, env):
        super(LiberoImageUpsideDownWrapper, self).__init__(env)

    def reset(self):
        obs = self.env.reset()
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["agentview_image"] = obs["agentview_image"][:, ::-1, :, :]  # (b, h, w, c)
        obs["robot0_eye_in_hand_image"] = obs["robot0_eye_in_hand_image"][:, ::-1, :, :]  # (b, h, w, c)
        return obs, reward, done, info


class LiberoObservationWrapper(ObservationWrapper):
    valid_obs_types = ["image"]

    def __init__(self, env, masks, cameras):
        super().__init__(env, masks, cameras)

    def _stack_obs(self, obs, axis=0):
        obs_dict = copy.deepcopy(obs)
        for t in self.valid_obs_types:
            obs_dict[t] = []
            for c in self.cameras:
                mod = obs[f"{c}_{t}"]
                obs_dict[t].append(mod)
            obs_dict[t] = np.stack(obs_dict[t], axis=1)  # (b, v, h, w, c)
        return obs_dict

class LiberoSuccessWrapper(Wrapper):
    def __init__(self, env):
        super(LiberoSuccessWrapper, self).__init__(env)
        self.success = None

    def reset(self):
        obs = self.env.reset()
        self.success = None
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.success is None:
            self.success = done
        else:
            assert len(self.success) == len(done)
            self.success = [self.success[i] or done[i] for i in range(len(done))]
        info["success"] = list(self.success)
        return obs, reward, done, info


def build_env(img_size, env_type, env_meta_fn=None, env_name=None, task_name=None,
              render_gpu_ids=-1, vec_env_num=1, seed=0, env_idx_start_end=None, **kwargs):
    """
    Build the rollout environment.
    Args:
        img_size: The resolution of the pixel observation.
        env_type: The type of environment benchmark. Choices: ["libero"].
        env_meta_fn: The path to robommimic meta data, which is used to specify the robomimic environments.
        env_name: The name to specify the environments.
        obs_types: The observation types in the returned obs dict in Robomimic
        render_gpu_ids:  The available GPU ids for rendering the images
        vec_env_num: The number of parallel environments
        seed: The random seed environment initialization.

    Returns:
        env: A gym-like environment.
    """
    if isinstance(img_size, Iterable):
        assert len(img_size) == 2
        img_h = img_size[0]
        img_w = img_size[1]
    else:
        img_h = img_w = img_size

    if env_type.lower() == "libero":
        if isinstance(render_gpu_ids, Iterable):
            render_gpu_ids = [int(i) for i in render_gpu_ids]
            gpu_id_for_each_env = render_gpu_ids * math.ceil(len(env_name) / len(render_gpu_ids))
            gpu_id_for_each_env = gpu_id_for_each_env[:len(env_name)]
        else:
            gpu_id_for_each_env = [render_gpu_ids] * len(env_name)

        if env_idx_start_end is not None:
            idx_start, idx_end = env_idx_start_end
        else:
            idx_start = 0
            idx_end = len(env_name)

        env_dict = OrderedDict()
        suite_to_task_embs = {}
        for env_idx in range(idx_start, idx_end):
            e_name, t_name, e_meta_fn, gpu_id = env_name[env_idx], task_name[env_idx], env_meta_fn[env_idx], gpu_id_for_each_env[env_idx]
            task_embedding = suite_to_task_embs.get(e_name, None)
            env, task_embs = make_libero_env(e_name, t_name, img_h, img_w, task_embedding=task_embedding,
                                             gpu_id=gpu_id, vec_env_num=vec_env_num, seed=seed)
            if e_name not in suite_to_task_embs:
                suite_to_task_embs[e_name] = task_embs
            env = LiberoImageUpsideDownWrapper(env)
            env = LiberoSuccessWrapper(env)
            with open(e_meta_fn) as f:
                env_meta = json.load(f)
            cameras = env_meta['env_kwargs']["camera_names"]
            cameras.sort()
            env = LiberoObservationWrapper(env, masks=None, cameras=cameras)
            env_dict[f"{e_name}/{t_name}"] = (env_idx, env)
        env = env_dict
    else:
        raise ValueError(f"Environment {env_type} is not supported!")
    return env
