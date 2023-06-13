#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
name=am

# export CUDA_VISIBLE_DEVICES=1
# name=am-ppo

# export CUDA_VISIBLE_DEVICES=2
# name=am-critic

for seed in 1234 1235 1236 1237 1238; do
    for reward_type in "minmax" "meansum"; do
        python run.py experiment="mdpp/$name" \
            seed="$seed" \
            env.reward_type="$reward_type"  \
            logger.wandb.name="$name-$reward_type"
    done
done
