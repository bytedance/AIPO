# AIPO: Improving Training Objective for Iterative Preference Optimization

✨This repository contains the code and models for our
paper [AIPO: Improving Training Objective for Iterative Preference Optimization](https://).

## News

- **[2024.09]** Initial release.

## Contents

- [Install Requirements](#install-requirements)
- [Training](#training)
    - [Basics for Training](#basics-for-training)
    - [Step 1. Install Requirements](#step-1-install-requirements)
    - [Step 2. Setup for Distributed Training](#step-2-setup-for-distributed-training)
    - [Step 3. Run Training Scripts](#step-3-run-training-scripts)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [License](#license)


## Install Requirements

The dependencies are listed in [pyproject.toml](pyproject.toml). To install all the requirements, run:

```shell
pip3 install -e .
# Optional: install additional dependencies for evaluation, testing, and analysis.
pip3 install -e '.[eval,test,tools]'
```

## Training

### Basics for Training

We use Hydra to organize experiments. To run experiments, use the main entry point:

```shell
# Note: `alignment_run` == `python3 -m alignment.run_train`.
# IMPORTANT: The `-m` option ensures our custom launcher is applied correctly by Hydra.
alignment_run -m +experiments=recipes/<config> ...
```

Iterative preference optimization consists of multiple phases, including instruction creation, response generation, and response ranking.
To support this training pipeline, we start from a set of bases and then implement each stage separately in `alignment/runner/stages`.
The default config for each stages is listed in `configs/experiments/stages`.
By combining different stages, we build the recipes for training, which are listed in [`configs/experiments/recipes`](configs/experiments/recipes). We provide the YAML configs along with shell scripts for running the training recipes.
An example script for running the iterative training process can be found [here](configs/experiments/recipes/ultrafeedback-self_instruct-pair_rm/run.sh).

### Step 1. Install Requirements

Follow the steps mentioned above to install the requirements.

### Step 2. Setup for Distributed Training

We create a resolver for obtaining the distributed config universally and automatically for each training stage through [OmegaConf variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#id25). Our custom resolver can be found at [`distributed_resolver.py`](alignment/utils/distributed_resolver.py).
To run our experiments, first set up the environment variables correctly on each machine:

```shell
# Recommend to add it to your training scripts or .rc file.
export DIST_NPROC_PER_NODE=8        # 8 GPUs per node
export DIST_NNODES=1                # 1 node for training on single machine
export DIST_NODE_RANK=0             # Rank of the current node (starting from 0)
export DIST_MASTER_ADDR=127.0.0.1   # IP address for the master node
export DIST_MASTER_PORT=29500       # Port for the master node
```

These variables are used to:

1. Set up sharding in data generation.
2. Set up distributed training for PyTorch during the training phase, provided by our [`hydra-torchrun-launcher`](https://github.com/acherstyx/hydra-torchrun-launcher).

To support training on multiple nodes, you must have shared storage on each machine and support concurrent read & write access (e.g. NFS).
It is recommended to first test on a single machine before moving on to multi-node distributed training.

### Step 3. Run training scripts

Run the training scripts:

```shell
# Options
export MAX_ITER=14
export LOG_ROOT="./log"
export PROJECT_NAME="alignment/iterative"
export EXPERIMENT_NAME="ultrafeedback-self_instruct-pair_rm"
export SELF_INSTRUCT_STAGE_ADDITIONAL_ARGS=""
export GENERATE_RESPONSES_STAGE_ADDITIONAL_ARGS=""
export PAIR_RM_STAGE_ADDITIONAL_ARGS=""
export DPO_STAGE_ADDITIONAL_ARGS="runner.training_args.beta=0.1" # Optional, overrides default config

# Execute
bash configs/experiments/recipes/ultrafeedback-self_instruct-pair_rm/run.sh
```


### Evaluation

Please refer to the original repositories for evaluation:

- [MT-Bench](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md)
- [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval)
- [Arena-Hard](https://github.com/lm-sys/arena-hard-auto)

We also provide our modified scripts for generating model output for evaluation in [`alignment/tools/evaluation`](alignment/tools/evaluation), which use vLLM for tensor parallel and faster generation.

## Citation

```text
@article{
    title={{AIPO}: Improving Training Objective for Iterative Preference Optimization},
    author={},
    journal={},
    year={2024}
}
```

## License

[![Code License](https://img.shields.io/badge/Code%20License-MIT-yellow.svg)](LICENSE)
[![Weight License](https://img.shields.io/badge/Weight%20License-CC%20By%20NC%204.0-red)](WEIGHT_LICENSE)

The weights of checkpoints are licensed under CC BY-NC 4.0 for non-commercial use. The codebase is licensed under MIT. 
This project utilizes certain datasets and checkpoints that are subject to their respective original licenses. 
Users must comply with all terms and conditions of these original licenses. 
The content produced by any version of AIPO is influenced by uncontrollable variables such as randomness, and therefore, the accuracy of the output cannot be guaranteed by this project. 
This project does not accept any legal liability for the content of the model output, nor does it assume responsibility for any losses incurred due to the use of associated resources and output results.
