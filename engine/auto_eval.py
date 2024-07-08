import os
import time
import glob
import argparse
import wandb
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Import necessary modules from the existing evaluation code
from engine.eval_mv_bc import evaluate, setup, get_ckp_name, save_success_rate
from lightning.fabric import Fabric
from omegaconf import OmegaConf

class CheckpointHandler(FileSystemEventHandler):
    def __init__(self, cfg, fabric, wandb_run, checkpoint_dir):
        self.cfg = cfg
        self.fabric = fabric
        self.wandb_run = wandb_run
        self.checkpoint_dir = checkpoint_dir
        self.evaluated_checkpoints = set()

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.ckpt'):
            self.evaluate_checkpoint(event.src_path)

    def evaluate_checkpoint(self, checkpoint_path):
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
            self.wandb_run.log({f"eval/video_{os.path.basename(video_file)}": wandb.Video(video_file)})

        # Log a summary of the evaluation 
        self.wandb_run.log({
            "epoch": get_ckp_name(checkpoint_path),
            "success_env_avg": results.get("rollout/success_env_avg", 0),
        })

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", required=True, help="Directory to monitor for new checkpoints")
    parser.add_argument("--wandb-run-id", required=True, help="Existing wandb run ID")
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.checkpoint_dir + "/config.yaml")

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

    # Create checkpoint handler
    handler = CheckpointHandler(cfg, fabric, wandb_run, args.checkpoint_dir)

    # Set up directory observer
    observer = Observer()
    observer.schedule(handler, args.checkpoint_dir, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

    wandb_run.finish()

if __name__ == "__main__":
    main()