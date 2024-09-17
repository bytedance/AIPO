#!/bin/bash

# iter 0

alignment_run -m \
  +experiments=recipes/ultrafeedback-pair_rm/iter_0_stage_1_generate_responses \
  dataset.n_samples=64 || exit 1
alignment_run -m \
  +experiments=recipes/ultrafeedback-pair_rm/iter_0_stage_2_pair_rm || exit 1
alignment_run -m \
  +experiments=recipes/ultrafeedback-pair_rm/iter_0_stage_3_dpo \
  runner.training_args.logging_steps=1 \
  runner.training_args.gradient_checkpointing=True || exit 1

# iter 1~2

for ITER in 1 2; do
  export ITERATION=${ITER}

  alignment_run -m \
    +experiments=recipes/ultrafeedback-pair_rm/iter_n_stage_1_generate_responses \
    dataset.n_samples=64 || exit 1
  alignment_run -m \
    +experiments=recipes/ultrafeedback-pair_rm/iter_n_stage_2_pair_rm || exit 1
  alignment_run -m \
    +experiments=recipes/ultrafeedback-pair_rm/iter_n_stage_3_dpo \
    runner.training_args.logging_steps=1 \
    runner.training_args.gradient_checkpointing=True || exit 1
done
