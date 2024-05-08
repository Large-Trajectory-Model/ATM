import torch

import os
import time
import numpy as np
from copy import deepcopy
from easydict import EasyDict
from collections import deque
from functools import partial

from libero import benchmark, get_libero_path
from libero.envs import OffScreenRenderEnv, DummyVectorEnv, SubprocVecEnv
from libero.envs.env_wrapper import ControlEnv
from robosuite.wrappers import Wrapper


def merge_dict(dict_obj):
    merged_dict = {}
    for k in dict_obj[0].keys():
        merged_dict[k] = np.stack([d[k] for d in dict_obj], axis=0)
    return merged_dict


class StackDummyVectorEnv(DummyVectorEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self, id=None):
        obs = super().reset(id=id)
        return merge_dict(obs)

    def step(self, action: np.ndarray, id=None,):
        obs, reward, done, info = super().step(action, id)
        return merge_dict(obs), reward, done, merge_dict(info)


class StackSubprocVectorEnv(SubprocVecEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        obs = super().reset()
        return merge_dict(obs)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        return merge_dict(obs), reward, done, merge_dict(info)


class LiberoTaskEmbWrapper(Wrapper):
    """ Wrapper to add task embeddings to the returned info """
    def __init__(self, env, task_emb):
        super().__init__(env)
        self.task_emb = task_emb

    def reset(self):
        obs = self.env.reset()
        obs["task_emb"] = self.task_emb
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs["task_emb"] = self.task_emb
        return obs, reward, done, info


class LiberoResetWrapper(Wrapper):
    """ Wrap the complex state initialization process in LIBERO """
    def __init__(self, env, init_states):
        super().__init__(env)
        assert isinstance(self.env, ControlEnv)
        self.init_states = init_states
        self.reset_times = 0

    def reset(self):
        _ = self.env.reset()
        obs = self.env.set_init_state(self.init_states[self.reset_times])

        # dummy actions all zeros for initial physics simulation
        dummy = np.zeros(7)
        dummy[-1] = -1.0  # set the last action to -1 to open the gripper
        for _ in range(5):
            obs, _, _, _ = self.env.step(dummy)

        self.reset_times += 1
        if self.reset_times == len(self.init_states):
            self.reset_times = 0
        return obs

    def seed(self, seed):
        self.env.seed(seed)


def make_libero_env(task_suite_name, task_name, img_h, img_w, task_embedding=None, gpu_id=-1, vec_env_num=1, seed=0):
    """
    Build a LIBERO environment according to the task suite name and task name.
    Args:
        task_suite_name: libero_10, libero_90, libero_spatial, libero_object or libero_goal.
        task_name: e.g., "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate"
        task_embedding: the BERT embedding of the task descriptions. If None, will get from BERT model (but too slow).
    Returns:

    """
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    # set the task embeddings
    cfg = EasyDict({
        "task_embedding_format": "bert",
        "task_embedding_one_hot_offset": 1,
        "data": {"max_word_len": 25},
        "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
    })  # hardcode the config to get task embeddings according to original Libero code
    descriptions = [task.language for task in task_suite.tasks]
    if task_embedding is None:
        task_embedding_map = np.load(os.path.join(get_libero_path("task_embeddings"), "task_emb_bert.npy"), allow_pickle=True).item()
        task_embs = torch.from_numpy(np.stack([task_embedding_map[des] for des in descriptions]))
    else:
        task_embs = task_embedding
    task_suite.set_task_embs(task_embs)

    # retrieve a specific task
    task_id = task_suite.get_task_id(task_name)
    task = task_suite.get_task_from_name(task_name)
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    print(f"[info] retrieving task {task_name} from suite {task_suite_name}, the " + \
          f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": img_h,
        "camera_widths": img_w,
        "render_gpu_device_id": gpu_id
    }

    init_states = task_suite.get_task_init_states(task_id)
    assert len(init_states) % vec_env_num == 0, "error: the number of initial states must be divisible by the number of envs"
    num_states_per_env = len(init_states) // vec_env_num
    def env_func(env_idx):
        base_env = OffScreenRenderEnv(**env_args)
        base_env = LiberoResetWrapper(base_env, init_states=init_states[env_idx*num_states_per_env:(env_idx+1)*num_states_per_env])
        base_env = LiberoTaskEmbWrapper(base_env, task_emb=task_suite.get_task_emb(task_id))
        base_env.seed(seed)
        return base_env

    env_created = False
    count = 0
    env = None
    while not env_created and count < 5:
        try:
            if vec_env_num == 1:
                env = StackDummyVectorEnv([partial(env_func, env_idx=i) for i in range(vec_env_num)])
            else:
                env = StackSubprocVectorEnv([partial(env_func, env_idx=i) for i in range(vec_env_num)])
            env_created = True
        except:
            time.sleep(5)
            count += 1
    if count >= 5:
        raise Exception("Failed to create environment")

    return env, task_embs
