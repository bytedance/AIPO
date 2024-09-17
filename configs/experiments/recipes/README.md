# Recipes

This directory contains recipes for iterative preference optimization.

To run a recipe, use the `run.sh` script present in each subdirectory. For example:

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

For debugging, use the `debug.sh` script located in each subdirectory. Optionally, you can run the `debug.sh` script in the current directory to execute all the `debug.sh` scripts in the subdirectories.