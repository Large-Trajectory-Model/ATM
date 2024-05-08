import os
import argparse
from glob import glob


# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_100"], 
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
args = parser.parse_args()

# training configs
CONFIG_NAME = "libero_track_transformer"

gpu_ids = [0, 1, 2, 3]

root_dir = "./data/atm_libero/"
suite_name = args.suite

# setup number of epoches and dataset path
if suite_name == "libero_100":
    EPOCH = 301
    train_dataset_list = glob(os.path.join(root_dir, "libero_90/*/train/")) + glob(os.path.join(root_dir, "libero_10/*/train/"))
    val1_dataset_list = glob(os.path.join(root_dir, "libero_90/*/val/")) + glob(os.path.join(root_dir, "libero_10/*/val/"))
else:
    EPOCH = 1001
    train_dataset_list = glob(os.path.join(root_dir, f"{suite_name}/*/train/"))
    val1_dataset_list = glob(os.path.join(root_dir, f"{suite_name}/*/val/"))

command = (f'python -m engine.train_track_transformer --config-name={CONFIG_NAME} '
           f'train_gpus="{gpu_ids}" '
           f'experiment={CONFIG_NAME}_{suite_name.replace("_", "-")}_ep{EPOCH} '
           f'epochs={EPOCH} '
           f'train_dataset="{train_dataset_list}" val_dataset="{val1_dataset_list}" ')

os.system(command)
