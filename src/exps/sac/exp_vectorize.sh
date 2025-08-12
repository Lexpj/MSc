#!/bin/bash

steps=1000008
rep=5
time=24:00:00
par=cpu-medium
mem=4G
env=HalfCheetah-v4

bash job.sh \
    --fw=SB3 \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/SB3_sac_defaultmujoco.yml

bash job.sh \
    --fw=SB3 \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/CleanRL_sac_defaultmujoco.yml

bash job.sh \
    --fw=CleanRL \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/SB3_sac_defaultmujoco.yml

bash job.sh \
    --fw=CleanRL \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/CleanRL_sac_defaultmujoco.yml

bash job.sh \
    --fw=SB3 \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/SB3_sac_defaultmujoco_envs.yml

bash job.sh \
    --fw=SB3 \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/CleanRL_sac_defaultmujoco_envs.yml

bash job.sh \
    --fw=CleanRL \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/SB3_sac_defaultmujoco_envs.yml

bash job.sh \
    --fw=CleanRL \
    --alg=sac \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps=hps/CleanRL_sac_defaultmujoco_envs.yml