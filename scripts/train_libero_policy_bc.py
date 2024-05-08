import os
import argparse


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# default track transformer path
DEFAULT_TRACK_TRANSFORMERS = {
    "libero_spatial": "./results/track_transformer/libero_track_transformer_libero-spatial/",
    "libero_object": "./results/track_transformer/libero_track_transformer_libero-object/",
    "libero_goal": "./results/track_transformer/libero_track_transformer_libero-goal/",
    "libero_10": "./results/track_transformer/libero_track_transformer_libero-100/",
}

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], 
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_vilt"

train_gpu_ids = [0, 1, 2, 3]
NUM_DEMOS = 10

root_dir = "./data/atm_libero/"
suite_name = args.suite
task_dir_list = os.listdir(os.path.join(root_dir, suite_name))
task_dir_list.sort()

# dataset
train_path_list = [f"{root_dir}/{suite_name}/{task_dir}/bc_train_{NUM_DEMOS}" for task_dir in task_dir_list]
val_path_list = [f"{root_dir}/{suite_name}/{task_dir}/val" for task_dir in task_dir_list]

track_fn = DEFAULT_TRACK_TRANSFORMERS[suite_name]  # just a placeholder, is not used when training vanilla BC

for seed in range(3):
    commond = (f'python -m engine.train_bc --config-name={CONFIG_NAME} train_gpus="{train_gpu_ids}" '
                f'experiment=bc-policy_{suite_name.replace("_", "-")}_demo{NUM_DEMOS} '
                f'train_dataset="{train_path_list}" val_dataset="{val_path_list}" '
                f'model_cfg.track_cfg.track_fn={track_fn} '
                f'model_cfg.track_cfg.use_zero_track=True '
                f'model_cfg.spatial_transformer_cfg.use_language_token=True '
                f'model_cfg.temporal_transformer_cfg.use_language_token=True '
                f'seed={seed} ')

    os.system(commond)
