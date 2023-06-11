#!/bin/bash

ENV=mdpp
EXP_NAME='am-critic'
export CUDA_VISIBLE_DEVICES=7 # change device id
MAX_EPOCHS=100
PROJECT_NAME=rl4co-mdpp
lr='1e-5'
wd='1e-5'

for seed in 1234, 1235, 1236;
    do

    if [ $EXP_NAME == 'am-critic' ]; then
        # AM-CRITIC
        python run.py experiment=${ENV}/am-critic \
        seed=$seed \
        logger.wandb.name=am-critic-${ENV} \
        logger.wandb.project=${PROJECT_NAME} \
        data.train_size=1000 \
        data.batch_size=16 \
        +data.val_batch_size=100 \
        train.optimizer.lr=${lr} \
        train.optimizer.weight_decay=${wd} \
        trainer.max_epochs=${MAX_EPOCHS}

    elif [ $EXP_NAME == 'am' ]; then
        # AM ( we have x2 samples, so less samples per epoch
        python run.py experiment=${ENV}/am \
        seed=$seed \
        logger.wandb.name=am-${ENV} \
        logger.wandb.project=${PROJECT_NAME} \
        data.train_size=500 \
        data.batch_size=16 \
        +data.val_batch_size=100 \
        train.optimizer.lr=${lr} \
        train.optimizer.weight_decay=${wd} \
        trainer.max_epochs=${MAX_EPOCHS}

    elif [ $EXP_NAME == 'am-ppo' ]; then
        # AM-PPO
        python run.py experiment=${ENV}/am-ppo \
        seed=$seed \
        logger.wandb.name=am-ppo-${ENV} \
        logger.wandb.project=${PROJECT_NAME} \
        data.train_size=1000 \
        data.batch_size=16 \
        +data.val_batch_size=100 \
        train.optimizer.lr=${lr} \
        train.optimizer.weight_decay=${wd} \
        trainer.max_epochs=${MAX_EPOCHS}
    fi

done
