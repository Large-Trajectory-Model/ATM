import os
import argparse

# environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# input parameters
parser = argparse.ArgumentParser()
parser.add_argument("--suite", default="libero_goal", choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"], 
                    help="The name of the desired suite, where libero_10 is the alias of libero_long.")
parser.add_argument("--exp-dir", required=True, help="The path to the folder of trained policy.")
args = parser.parse_args()

# evaluation configs
train_gpu_ids = [0, 1, 2, 3]
env_gpu_ids = [4, 5, 6, 7]

root_dir = "./data/atm_libero"
suite_name = args.suite
task_dir_list = os.listdir(os.path.join(root_dir, suite_name))
task_dir_list.sort()

# environment
suite_name_list = [suite_name] * len(task_dir_list)
task_name_list = [task_dir.replace('_demo', '') for task_dir in task_dir_list]
env_meta_path_list = [f"{root_dir}/{suite_name}/{task_dir}/env_meta.json" for task_dir in task_dir_list]

exp_dir = args.exp_dir
command = (f'python -m engine.eval_mv_bc --config-dir={exp_dir} --config-name=config hydra.run.dir=/tmp '
            f'+save_path={exp_dir} '
            f'train_gpus="{train_gpu_ids}" '
            f'env_cfg.env_name="{suite_name_list}" env_cfg.task_name="{task_name_list}" env_cfg.env_meta_fn="{env_meta_path_list}" '
            f'env_cfg.render_gpu_ids="{env_gpu_ids}" env_cfg.vec_env_num=10 ')

os.system(command)
