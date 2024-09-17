#!/bin/bash

alignment_run -m \
  +experiments=recipes/ultrafeedback/iter_0_stage_1_dpo \
  dataset.split='train_prefs\[:100\]' \
  runner.training_args.logging_steps=1 \
  runner.training_args.gradient_checkpointing=True || exit 1


# iter 1~2

for ITER in 1 2; do
  export ITERATION=${ITER}

  alignment_run -m \
    +experiments=recipes/ultrafeedback/iter_n_stage_1_dpo \
    runner.training_args.logging_steps=1 \
    runner.training_args.gradient_checkpointing=True || exit 1
done
