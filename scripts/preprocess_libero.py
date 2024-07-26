import json
import os
from glob import glob

import click
import h5py
import numpy as np
import torch
from einops import rearrange
from natsort import natsorted
from tqdm import tqdm
from easydict import EasyDict
from PIL import Image
from transformers import AutoTokenizer, AutoModel
from hydra.utils import to_absolute_path

from atm.utils.flow_utils import sample_from_mask, sample_double_grid
from atm.utils.cotracker_utils import Visualizer

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

EXTRA_STATES_KEYS = ['gripper_states', 'joint_states', 'ee_ori', 'ee_pos', 'ee_states']


def get_task_name_from_file_name(file_name):
    name = file_name.replace('_demo', '')
    if name[0].isupper():  # LIBERO-10 and LIBERO-90
        if "SCENE10" in name:
            language = " ".join(name[name.find("SCENE") + 8 :].split("_"))
        else:
            language = " ".join(name[name.find("SCENE") + 7 :].split("_"))
    else:
        language = " ".join(name.split("_"))
    return language


def get_task_embs(cfg, descriptions):
    """
    Bert embeddings for task embeddings. Borrow from https://github.com/Lifelong-Robot-Learning/LIBERO/blob/f78abd68ee283de9f9be3c8f7e2a9ad60246e95c/libero/lifelong/utils.py#L152.
    """
    if cfg.task_embedding_format == "bert":
        tz = AutoTokenizer.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./data/bert")
        )
        model = AutoModel.from_pretrained(
            "bert-base-cased", cache_dir=to_absolute_path("./data/bert")
        )
        tokens = tz(
            text=descriptions,  # the sentence to be encoded
            add_special_tokens=True,  # Add [CLS] and [SEP]
            max_length=cfg.data.max_word_len,  # maximum length of a sentence
            padding="max_length",
            return_attention_mask=True,  # Generate the attention mask
            return_tensors="pt",  # ask the function to return PyTorch tensors
        )
        masks = tokens["attention_mask"]
        input_ids = tokens["input_ids"]
        task_embs = model(tokens["input_ids"], tokens["attention_mask"])[
            "pooler_output"
        ].detach()
    else:
        raise ValueError("Unsupported task embedding format")
    cfg.policy.language_encoder.network_kwargs.input_size = task_embs.shape[-1]
    return task_embs


def get_task_bert_embs(libero_root_dir):
    libero_h5_files = glob(os.path.join(libero_root_dir, "*/*.hdf5"))
    task_names = set([get_task_name_from_file_name(os.path.basename(file).split('.')[0]) for file in libero_h5_files])
    task_names = list(task_names)

    if not os.path.exists("libero/task_embedding_caches/task_emb_bert.npy"):
        # set the task embeddings
        cfg = EasyDict({
            "task_embedding_format": "bert",
            "task_embedding_one_hot_offset": 1,
            "data": {"max_word_len": 25},
            "policy": {"language_encoder": {"network_kwargs": {"input_size": 768}}}
        })  # hardcode the config to get task embeddings according to original Libero code

        task_embs = get_task_embs(cfg, task_names).cpu().numpy()

        task_name_to_emb = {task_names[i]: task_embs[i] for i in range(len(task_names))}

        os.makedirs("libero/task_embedding_caches/", exist_ok=True)
        np.save("libero/task_embedding_caches/task_emb_bert.npy", task_name_to_emb)
    else:
        task_name_to_emb = np.load("libero/task_embedding_caches/task_emb_bert.npy", allow_pickle=True).item()
    return task_name_to_emb


def track_and_remove(tracker, video, points, var_threshold=10.):
    B, T, C, H, W = video.shape
    pred_tracks, pred_vis = tracker(video, queries=points, backward_tracking=True) # [1, T, N, 2]

    var = torch.var(pred_tracks, dim=1)  # [1, N, 2]
    var = torch.sum(var, dim=-1)[0]  # List

    # get index of points with low variance
    idx = torch.where(var > var_threshold)[0]
    if len(idx) == 0:
        print(torch.max(var))
        assert len(idx) > 0, 'No points with low variance'

    new_points = points[:, idx].clone()

    # Repeat and sample
    rep = points.shape[1] // len(idx) + 1
    new_points = torch.tile(new_points, (1, rep, 1))
    new_points = new_points[:, :points.shape[1]]
    # Add 10 percent height and width as noise
    noise = torch.randn_like(new_points[:, :, 1:]) * 0.05 * H
    new_points[:, :, 1:] += noise

    # Track new points
    pred_tracks, pred_vis = tracker(video, queries=new_points, backward_tracking=True)

    return pred_tracks, pred_vis


def track_through_video(video, track_model, num_points=1000):
    T, C, H, W = video.shape

    video = torch.from_numpy(video).cuda().float()

    # sample random points
    points = sample_from_mask(np.ones((H, W, 1)) * 255, num_samples=num_points)
    points = torch.from_numpy(points).float().cuda()
    points = torch.cat([torch.randint_like(points[:, :1], 0, T), points], dim=-1).cuda()

    # sample grid points
    grid_points = sample_double_grid(7, device="cuda")
    grid_points[:, 0] = grid_points[:, 0] * H
    grid_points[:, 1] = grid_points[:, 1] * W
    grid_points = torch.cat([torch.randint_like(grid_points[:, :1], 0, T), grid_points], dim=-1).cuda()

    pred_tracks, pred_vis = track_and_remove(track_model, video[None], points[None])
    pred_grid_tracks, pred_grid_vis = track_and_remove(track_model, video[None], grid_points[None], var_threshold=0.)

    pred_tracks = torch.cat([pred_grid_tracks, pred_tracks], dim=2)
    pred_vis = torch.cat([pred_grid_vis, pred_vis], dim=2)
    return pred_tracks, pred_vis


def collect_states_from_demo(h5_file, image_save_dir, demos_group, demo_k, view_names, track_model, task_emb, num_points, visualizer, save_vis=False):
    actions = np.array(demos_group[demo_k]['actions'])
    root_grp = h5_file.create_group("root") if "root" not in h5_file else h5_file["root"]
    if "actions" not in root_grp:
        root_grp.create_dataset("actions", data=actions)

    if "extra_states" not in root_grp:
        extra_states_grp = root_grp.create_group("extra_states")
        for state_key in EXTRA_STATES_KEYS:
            extra_states_grp.create_dataset(state_key, data=np.array(demos_group[demo_k]['obs'][state_key]))

    if "task_emb_bert" not in root_grp:
        root_grp.create_dataset("task_emb_bert", data=task_emb)

    for view in view_names:
        rgb = np.array(demos_group[demo_k]['obs'][f'{view}_rgb'])
        rgb = rgb[:, ::-1, :, :].copy()  # The images in the raw Libero dataset is upsidedown, so we need to flip it
        rgb = rearrange(rgb, "t h w c -> t c h w")
        T, C, H, W = rgb.shape

        pred_tracks, pred_vis = track_through_video(rgb, track_model, num_points=num_points)

        if save_vis:
            visualizer.visualize(torch.from_numpy(rgb)[None], pred_tracks, pred_vis, filename=f"{demo_k}_{view}")

        # [1, T, N, 2], normalize coordinates to [0, 1] for in-picture coordinates
        pred_tracks[:, :, :, 0] /= W
        pred_tracks[:, :, :, 1] /= H

        # hierarchically save arrays under the view name
        view_grp = root_grp.create_group(view) if view not in root_grp else root_grp[view]
        if "video" not in view_grp:
            view_grp.create_dataset("video", data=rgb[None].astype(np.uint8))

        # we always update the tracks and vis when you run this script
        if "tracks" in view_grp:
            view_grp.__delitem__("tracks")
        if "vis" in view_grp:
            view_grp.__delitem__("vis")
        view_grp.create_dataset("tracks", data=pred_tracks.cpu().numpy())
        view_grp.create_dataset("vis", data=pred_vis.cpu().numpy())

        # save image pngs
        save_images(rearrange(rgb, "t c h w -> t h w c"), image_save_dir, view)


def save_images(video, image_dir, view_name):
    os.makedirs(image_dir, exist_ok=True)
    for idx, img in enumerate(video):
        Image.fromarray(img).save(os.path.join(image_dir, f"{view_name}_{idx}.png"))


def inital_save_h5(path, skip_exist):
    if os.path.exists(path):
        with h5py.File(path, 'r') as f:
            if ("agentview" in f["root"]) and ("eye_in_hand" in f["root"]) and skip_exist:
                return None

    f = h5py.File(path, 'w')
    return f


def get_attrs_and_view_names(demo_h5):
    """ Get preproception states from h5 file object. """
    attrs = json.loads(demo_h5.attrs['env_args'])
    views = attrs['env_kwargs']['camera_names']
    views.sort()

    views = [name.replace('robot0_', '') if name.endswith("eye_in_hand") else name for name in views]
    return attrs, views


def generate_data(source_h5_path, target_dir, track_model, task_emb, skip_exist):
    demos = h5py.File(source_h5_path, 'r')['data']
    demo_keys = natsorted(list(demos.keys()))
    attrs, views = get_attrs_and_view_names(demos)

    # save environment meta data
    with open(os.path.join(target_dir, 'env_meta.json'), 'w') as fp:
        json.dump(attrs, fp)

    # setup visualization class
    video_path = os.path.join(target_dir, 'videos')
    if not os.path.exists(video_path):
        os.makedirs(video_path, exist_ok=True)
    visualizer = Visualizer(save_dir=video_path, pad_value=0, fps=24)

    num_points = 1000
    with torch.no_grad():
        for idx in tqdm(range(len(demo_keys))):
            demo_k = demo_keys[idx]
            save_path = os.path.join(target_dir, f"{demo_k}.hdf5")
            h5_file_handle = inital_save_h5(save_path, skip_exist)
            image_save_dir = os.path.join(target_dir, "images", demo_k)

            if h5_file_handle is None:
                continue

            try:
                collect_states_from_demo(h5_file_handle, image_save_dir, demos, demo_k, views, track_model, task_emb, num_points, visualizer, save_vis=(idx%10==0))
                h5_file_handle.close()
                print(f"{save_path} is completed.")
            except Exception as e:
                print(f"Exception {e} when processing {save_path}")
                h5_file_handle.close()
                exit()


@click.command()
@click.option("--root", type=str, default="./data/libero/")
@click.option("--save", type=str, default="./data/atm_libero/")
@click.option("--suite", type=str, default="libero_spatial")
@click.option("--skip_exist", type=bool, default=False)
def main(root, save, suite, skip_exist):
    """
    root: str, root directory of original libero dataset
    save: str, target directory to save the preprocessed data
    suite: str, the name of assigned suite, [libero_spatial, libero_object, libero_goal, libero_10, libero_90]
    skip_exist: bool, whether to skip the existing preprocessed h5df file
    """
    suite_dir = os.path.join(root, suite)

    # setup cotracker
    cotracker = torch.hub.load(os.path.join(os.path.expanduser("~"), ".cache/torch/hub/facebookresearch_co-tracker_main/"), "cotracker2", source="local")
    cotracker = cotracker.eval().cuda()

    # load task name embeddings
    task_bert_embs_dict = get_task_bert_embs(root)

    for source_h5 in os.listdir(suite_dir):
        source_h5_path = os.path.join(suite_dir, source_h5)
        file_name = source_h5.split('.')[0]
        task_name = get_task_name_from_file_name(file_name)
 
        save_dir = os.path.join(save, suite, file_name)
        os.makedirs(save_dir, exist_ok=True)
        generate_data(source_h5_path, save_dir, cotracker, task_bert_embs_dict[task_name], skip_exist)


if __name__ == "__main__":
    main()
