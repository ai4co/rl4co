#!/bin/bash

RUN_CMD="python scripts/eval_methods_pareto.py"

# TSP20
SIZE=20
TASK="tsp${SIZE}"
GPU_ID=0
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/pomo-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/symnco-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-${TASK}-sm-xl" --gpu_id ${GPU_ID} --checkpoint "epoch_499.ckpt"
${RUN_CMD} --dir "saved_checkpoints/${TASK}/am-sm-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_499.ckpt"
# ${RUN_CMD} --dir "saved_checkpoints/${TASK}/ptrnet-${TASK}" --gpu_id ${GPU_ID} --checkpoint "epoch_099.ckpt"