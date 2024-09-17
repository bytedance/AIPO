#!/bin/bash

MAX_ITER=${MAX_ITER:-2}

##########
# iter 0 #
##########
export ITERATION=0
echo "=================================================="
echo "$(date) Iter ${ITERATION}"
echo "=================================================="

if python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${LOG_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/iter${ITERATION}/dpo')"; then
  echo "$(date) Skip iter ${ITERATION}"
else

  alignment_run -m \
    +experiments=recipes/ultrafeedback/iter_0_stage_1_dpo \
    log.root="${LOG_ROOT}" log.project_name="${PROJECT_NAME}" log.experiment_name="${EXPERIMENT_NAME}" \
    ${DPO_STAGE_ADDITIONAL_ARGS} || exit 1

fi

##########
# iter N #
##########
for ITER in $(seq 1 "$MAX_ITER"); do
  export ITERATION=${ITER}
  echo "=================================================="
  echo "$(date) Iter ${ITERATION}"
  echo "=================================================="

  if python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('${LOG_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/iter${ITERATION}/dpo')"; then
    echo "$(date) Skip iter ${ITERATION}"
    continue
  fi

  alignment_run -m \
    +experiments=recipes/ultrafeedback/iter_n_stage_1_dpo \
    log.root="${LOG_ROOT}" log.project_name="${PROJECT_NAME}" log.experiment_name="${EXPERIMENT_NAME}" \
    ${DPO_STAGE_ADDITIONAL_ARGS} || exit 1

done
