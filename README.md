# Any-point Trajectory Modeling for Policy Learning

[Chuan Wen](https://alvinwen428.github.io)\*,
[Xingyu Lin](https://xingyu-lin.github.io)\*,
[John So](https://www.johnrso.xyz/)\*,
[Kai Chen](https://ck-kai.github.io/),
[Dou Qi](https://www.cse.cuhk.edu.hk/~qdou/),
[Yang Gao](https://yang-gao.weebly.com/),
[Pieter Abbeel](https://people.eecs.berkeley.edu/~pabbeel/)

[paper](https://arxiv.org/abs/2401.00025) | [website](https://xingyu-lin.github.io/atm/)

**Robotics: Science and Systems (RSS) 2024**

![image](https://github.com/Large-Trajectory-Model/ATM/blob/main/doc/pull_figure.gif)

## Installation

```
git clone --recursive https://github.com/Large-Trajectory-Model/ATM.git

cd ATM/
conda env create -f environment.yml
conda activate atm

pip install -e third_party/robosuite/
pip install -e third_party/robomimic/
```

## Dataset Preprocessing

We first need to download the raw LIBERO datasets:

```
mkdir data
python -m scripts.download_libero_datasets
```

and then preprocess them with [Cotracker](https://arxiv.org/abs/2307.07635):

```
python -m scripts.preprocess_libero --suite libero_spatial
python -m scripts.preprocess_libero --suite libero_object
python -m scripts.preprocess_libero --suite libero_goal
python -m scripts.preprocess_libero --suite libero_10
python -m scripts.preprocess_libero --suite libero_90
```

After preprocessing, split the datasets into training and validation sets:

```
python -m scripts.split_libero_dataset
```

## Download Checkpoints

To reproduce the experimental results in our paper, we provide the checkpoints trained by us. Please download the zip file from [here](https://drive.google.com/file/d/1lG2hNG_-Etu2TG7XbGxOD40Gorj2DtLw/view?usp=sharing) and put it in the current folder. Then,

```
mkdir results
unzip -o atm_release_checkpoints.zip -d results/
rm atm_release_checkpoints.zip
```

## Training

As shown in Figure 2 in our paper, the training of our  Trajectory Modeling framework includes two stages: **Track Transformer Pretraining** and **Trajectory-guided Policy Training**.

### Stage 1: Track Transformer Pretraining

The Track Transformer training can be executed by this command, where SUITE_NAME can be *libero_spatial*, *libero_object*, *libero_goal*, or *libero_100*:

```
python -m scripts.train_libero_track_transformer --suite $SUITE_NAME
```

### Stage 2: Track-guided Policy Training

The vanilla BC baseline can be trained by the following command, where $SUITE_NAME can be *libero_spatial*, *libero_object*, *libero_goal*, or *libero_10* (i.e., LIBERO-Long in our paper):
```
python -m scripts.train_libero_policy_bc --suite $SUITE_NAME
```

Our Track-guided policy can be trained with:
```
python -m scripts.train_libero_policy_atm --suite $SUITE_NAME --tt $PATH_TO_TT
```
where $PATH_TO_TT is the path to the folder of Track Transformer pretrained in Stage 1. We have provided the pretrained checkpoints in `results/track_transformers/`. For example,
```
python -m scripts.train_libero_policy_atm --suite libero_spatial --tt results/track_transformer/libero_track_transformer_libero-spatial/
python -m scripts.train_libero_policy_atm --suite libero_object --tt results/track_transformer/libero_track_transformer_libero-object/
python -m scripts.train_libero_policy_atm --suite libero_goal --tt results/track_transformer/libero_track_transformer_libero-goal/
python -m scripts.train_libero_policy_atm --suite libero_10 --tt results/track_transformer/libero_track_transformer_libero-100/
```

## Evaluation

The evaluation can be executed by this command, where $SUITE_NAME is the desired suite name and $PATH_TO_EXP is the path to the your trained policy folder in `results/policy/`. The success rate and evaluation videos will be saved in `$PATH_TO_EXP/eval_results/`.

```
python -m scripts.eval_libero_policy --suite $SUITE_NAME --exp-dir $PATH_TO_EXP
```

For example, you can evaluate the provided checkpoints by:

```
python -m scripts.eval_libero_policy --suite libero_spatial --exp-dir results/policy/atm-policy_libero-spatial_demo10
python -m scripts.eval_libero_policy --suite libero_object --exp-dir results/policy/atm-policy_libero-object_demo10
python -m scripts.eval_libero_policy --suite libero_goal --exp-dir results/policy/atm-policy_libero-goal_demo10
python -m scripts.eval_libero_policy --suite libero_10 --exp-dir results/policy/atm-policy_libero-10_demo10
```

## Citation

If you find our codebase is useful for your research, please cite our paper with this bibtex:

```
@article{wen2023atm,
  title={Any-point trajectory modeling for policy learning},
  author={Wen, Chuan and Lin, Xingyu and So, John and Chen, Kai and Dou, Qi and Gao, Yang and Abbeel, Pieter},
  journal={arXiv preprint arXiv:2401.00025},
  year={2023}
}
```
