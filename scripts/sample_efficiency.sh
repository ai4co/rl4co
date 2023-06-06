!#/bin/bash


##################################
# RUN FOR ALL 
# ONLY THING TO CHANGE
# NUM_NODES: 20, 50, 100
# ENV: tsp, cvrp
# EXP_NAME: am-critic, am, pomo, symnco

NUM_NODES=20
ENV=tsp
EXP_NAME='am-critic'
export CUDA_VISIBLE_DEVICES=0 # change device id

##################################




# 1 epoch: 50k samples for all models!!
# 40 epochs: 50k x 40 = 2M samples
# we want to normalize the number of samples AND gradient steps
for seed in 1234, 1235, 1236;
    do

    if [ $EXP_NAME == 'am-critic' ]; then
        # AM-CRITIC
        # 20, 50, 100 nodes
        python run.py experiment=${ENV}/am-critic \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=am-critic-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency \
        data.train_size=50000 \
        data.batch_size=500 \
        trainer.max_epochs=40
    elif [ $EXP_NAME == 'am' ]; then
        # AM ( we have x2 samples, so less samples per epoch
        # 20, 50, 100 nodes 
        python run.py experiment=${ENV}/am \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=am-critic-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency \
        data.train_size=25000 \
        data.batch_size=250 \
        trainer.max_epochs=40
    elif [ $EXP_NAME == 'pomo' ]; then

        if [ $NUM_NODES == 20 ]; then
            # POMO (20 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency \
            data.train_size=50000 \
            data.batch_size=25 \
            trainer.max_epochs=40
        elif [ $NUM_NODES == 50 ]; then
            # POMO (50 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency \
            data.train_size=50000 \
            data.batch_size=10 \
            trainer.max_epochs=40
        elif [ $NUM_NODES == 100 ]; then
            # POMO (100 nodes)
            python run.py experiment=${ENV}/pomo \
            env.num_loc=${NUM_NODES} \
            seed=$seed \
            logger.wandb.name=pomo-${ENV}-${NUM_NODES} \
            logger.wandb.project=rl4co-sample-efficiency \
            data.train_size=50000 \
            data.batch_size=5 \
            trainer.max_epochs=40
        fi

    elif [ $EXP_NAME == 'symnco' ]; then
        # SYMNCO (by default, 10 augment)
        # 20, 50, 100 nodes
        python run.py experiment=${ENV}/symnco \
        env.num_loc=${NUM_NODES} \
        seed=$seed \
        logger.wandb.name=symnco-${ENV}-${NUM_NODES} \
        logger.wandb.project=rl4co-sample-efficiency \
        data.train_size=50000 \
        data.batch_size=50 \
        trainer.max_epochs=40
    fi
done
