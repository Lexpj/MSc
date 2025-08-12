#!/bin/bash

env=HalfCheetah-v4
steps=3000005
rep=5
time=12:00:00
par=cpu-medium
mem=4G

bash job.sh \
    --fw=SB3 \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/SB3_ppo_defaultmujoco_wrapped.yml"
    
bash job.sh \
    --fw=SB3 \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/CleanRL_ppo_defaultmujoco_wrapped.yml"

bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/CleanRL_ppo_defaultmujoco_wrapped.yml"
    
bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/SB3_ppo_defaultmujoco_wrapped.yml"

    



