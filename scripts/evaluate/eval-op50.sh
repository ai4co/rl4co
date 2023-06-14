#!/bin/bash

RUN_CMD="python scripts/eval_methods.py"

################
# CONFIGS
ENV="op"
SIZE=50
TASK="${ENV}${SIZE}"
GPU_ID=3
################

${RUN_CMD} --dir "saved_checkpoints2/${TASK}/am-${TASK}" --gpu_id ${GPU_ID} --checkpoint "last.ckpt"
${RUN_CMD} --dir "saved_checkpoints2/${TASK}/pomo-${TASK}" --gpu_id ${GPU_ID} --checkpoint "last.ckpt"
${RUN_CMD} --dir "saved_checkpoints2/${TASK}/symnco-${TASK}" --gpu_id ${GPU_ID} --checkpoint "last.ckpt"
${RUN_CMD} --dir "saved_checkpoints2/${TASK}/am-${TASK}-sm-xl" --gpu_id ${GPU_ID} --checkpoint "last.ckpt"
