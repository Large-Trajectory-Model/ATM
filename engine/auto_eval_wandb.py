import os
import time
import glob
import argparse
import wandb
from omegaconf import OmegaConf

# Import necessary modules from the existing evaluation code
from engine.eval_mv_bc import evaluate, setup, get_ckp_name, save_success_rate
from lightning.fabric import Fabric

from atm.utils.train_utils import pretty_print_cfg

class CheckpointEvaluator:
    def __init__(self, cfg, fabric, wandb_run, checkpoint_dir):
        self.cfg = cfg
        self.fabric = fabric
        self.wandb_run = wandb_run
        self.checkpoint_dir = checkpoint_dir
        self.evaluated_checkpoints = set()

    def evaluate_checkpoint(self, checkpoint_path):
        # Ensure checkpoint_path is a string
        if hasattr(checkpoint_path, 'name'):
            checkpoint_path = checkpoint_path.name

        # If is model_best, then skip
        if "model_best" in checkpoint_path:
            return

        if checkpoint_path in self.evaluated_checkpoints:
            return

        print(f"Evaluating new checkpoint: {checkpoint_path}")
        self.evaluated_checkpoints.add(checkpoint_path)

        # Update the checkpoint path in the config
        self.cfg.model_cfg.load_path = checkpoint_path

        # Construct the video save directory
        checkpoint_name = os.path.basename(checkpoint_path).split('.')[0]
        video_save_dir = os.path.join(self.checkpoint_dir, "eval_results", "video_libero_spatial", f"epoch_{get_ckp_name(checkpoint_path)}")

        # Modify configuration
        root_dir = "./data/atm_libero"
        suite_name = "libero_spatial"
        task_dir_list = os.listdir(os.path.join(root_dir, suite_name))
        task_dir_list.sort()

        suite_name_list = [suite_name] * len(task_dir_list)
        task_name_list = [task_dir.replace('_demo', '') for task_dir in task_dir_list]
        env_meta_path_list = [f"{root_dir}/{suite_name}/{task_dir}/env_meta.json" for task_dir in task_dir_list]

        self.cfg.env_cfg.env_name = suite_name_list
        self.cfg.env_cfg.task_name = task_name_list
        self.cfg.env_cfg.env_meta_fn = env_meta_path_list
        
        # Run evaluation
        results = evaluate(self.fabric, self.cfg, checkpoint_path,
                           video_save_dir=video_save_dir,
                           num_env_rollouts=self.cfg.get("num_env_rollouts", 20))

        success_metrics = {k: v for k, v in results.items() if k.startswith("rollout/success_env")}
        save_success_rate(get_ckp_name(checkpoint_path), success_metrics, os.path.join(self.checkpoint_dir, "eval_results", f"summary_{suite_name}.csv"))

        # Log results to wandb
        video_files = glob.glob(os.path.join(video_save_dir, "*.mp4"))
        for video_file in video_files:
            print("logging video:", video_file)
            self.wandb_run.log({f"eval/video_{os.path.basename(video_file)}": wandb.Video(video_file)})

        # Log evaluation results
        self.wandb_run.log({
            "eval/epoch": get_ckp_name(checkpoint_path),
            "eval/success_env_avg": results.get("rollout/success_env_avg", 0),
            "eval/success_env0": results.get("rollout/success_env0", 0),
            "eval/success_env1": results.get("rollout/success_env1", 0),
            "eval/success_env2": results.get("rollout/success_env2", 0),
            "eavl/success_env3": results.get("rollout/success_env3", 0),
            "eval/success_env4": results.get("rollout/success_env4", 0),
            "eval/success_env5": results.get("rollout/success_env5", 0),
            "eval/success_env6": results.get("rollout/success_env6", 0),
            "eval/success_env7": results.get("rollout/success_env7", 0),
            "eval/success_env8": results.get("rollout/success_env8", 0),
            "eval/success_env9": results.get("rollout/success_env9", 0),
        })

        print(f"Finished evaluating checkpoint: {checkpoint_path}")

def get_wandb_config(run):
    wandb_config = run.config
    
    # Convert wandb config to a plain dictionary
    config_dict = dict(wandb_config)
    
    # Recursively convert any remaining wandb.Config objects to dictionaries
    def convert_config(item):
        if isinstance(item, dict):
            return {k: convert_config(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_config(v) for v in item]
        elif hasattr(item, '_items'):  # Check if it's a wandb.Config object
            return convert_config(dict(item._items))
        else:
            return item
    
    config_dict = convert_config(config_dict)
    
    # Create OmegaConf object from the cleaned dictionary
    cfg = OmegaConf.create(config_dict)

    pretty_print_cfg(cfg)
    
    return cfg
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to save downloaded checkpoints")
    parser.add_argument("--wandb-run-id", required=True, help="Existing wandb run ID")
    args = parser.parse_args()

    # Initialize wandb
    api = wandb.Api()
    project = "atm_policy"
    run = api.run(f"11485-26/{project}/{args.wandb_run_id}")

    cfg = get_wandb_config(run)

    # Resume existing wandb run
    wandb_run = wandb.init(
        id=args.wandb_run_id,
        project=cfg.wandb.project,
        entity="11485-26",
        resume="must",
    )

    # Setup Fabric
    fabric = Fabric(accelerator="cuda", devices=list(cfg.train_gpus), strategy="ddp")
    fabric.launch()

    # Setup evaluation
    setup(cfg)

    # Create checkpoint evaluator
    evaluator = CheckpointEvaluator(cfg, fabric, wandb_run, args.checkpoint_dir)

    evaluated_checkpoints = set()

    try:
        while True:
            # Query for new checkpoints
            files = run.files()
            checkpoint_artifacts = [f for f in files if f.name.endswith(".ckpt")]
            
            for artifact in checkpoint_artifacts:
                if artifact.name not in evaluated_checkpoints:
                    print(f"New checkpoint found: {artifact.name}")
                    # Download the checkpoint
                    checkpoint_path = artifact.download(root=args.checkpoint_dir, replace=True)

                    if isinstance(checkpoint_path, list):
                        checkpoint_path = checkpoint_path[0]

                    # Evaluate the checkpoint
                    evaluator.evaluate_checkpoint(checkpoint_path)
                    evaluated_checkpoints.add(artifact.name)
            
            # Wait for a while before checking again
            time.sleep(5000)  
    except KeyboardInterrupt:
        print("Evaluation stopped by user.")

    wandb_run.finish()

if __name__ == "__main__":
    main()