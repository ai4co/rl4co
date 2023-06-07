#!/bin/bash


##################################
# RUN FOR ALL 
# NUM_NODES: 20, 50, 100
# ENV: tsp, cvrp
# EXP_NAME: am-critic, am, pomo, symnco, am-ppo

# 1 epoch: 20k samples for all models!!
# 100 epochs: 20k x 100 = 2M samples
# we want to normalize the number of samples AND gradient steps

NUM_NODES=50
ENV="tsp"
EXP_NAME='am'
export CUDA_VISIBLE_DEVICES=0 # change device id

# DO NOT CHANGE
MAX_EPOCHS=100 

##################################


for seed in 1234, 1235, 1236;
    do
    if [ $EXP_NAME == 'am-critic' ]; then
        # AM-CRITIC
        # 20, 50, 100 nodes
        python run.py experiment=${ENV}/am-critic \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=am-critic-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency2 \
        data.train_size=20000 \
        data.batch_size=500 \
        trainer.max_epochs=${MAX_EPOCHS}
    elif [ $EXP_NAME == 'am' ]; then
        # AM ( we have x2 samples, so less samples per epoch
        # 20, 50, 100 nodes 
        python run.py experiment=transfer/am \
        transfer.target.size=${NUM_NODES} \
        seed=$seed \
        data.train_size=10000 \
        data.batch_size=250 \
        trainer.max_epochs=${MAX_EPOCHS}
    elif [ $EXP_NAME == 'pomo' ]; then

        if [ $NUM_NODES == 20 ]; then
            # POMO (20 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency2 \
            data.train_size=1000 \
            data.batch_size=25 \
            trainer.max_epochs=${MAX_EPOCHS}
        elif [ $NUM_NODES == 50 ]; then
            # POMO (50 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency2 \
            data.train_size=400 \
            data.batch_size=10 \
            trainer.max_epochs=${MAX_EPOCHS}
        elif [ $NUM_NODES == 100 ]; then
            # POMO (100 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency2 \
            data.train_size=200 \
            data.batch_size=5 \
            trainer.max_epochs=${MAX_EPOCHS}
        fi

    elif [ $EXP_NAME == 'symnco' ]; then
        # SYMNCO (by default, 10 augment)
        # 20, 50, 100 nodes
        python run.py experiment=${ENV}/symnco \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=symnco-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency2 \
        data.train_size=2000 \
        data.batch_size=50 \
        trainer.max_epochs=${MAX_EPOCHS}
    
    elif [ $EXP_NAME == 'am-ppo' ]; then
        # AM-PPO
        # 20, 50, 100 nodes
        python run.py experiment=${ENV}/am-ppo \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=am-critic-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency2 \
        data.train_size=20000 \
        data.batch_size=500 \
        trainer.devices=1 \
        trainer.max_epochs=${MAX_EPOCHS}
    fi

done
