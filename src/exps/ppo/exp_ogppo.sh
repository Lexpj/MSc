#!/bin/bash

envs=("InvertedDoublePendulum-v4" "InvertedPendulum-v4" "Reacher-v4" "Swimmer-v4" "Walker2d-v4" "HalfCheetah-v4" "Hopper-v4" "Ant-v4")
fws=("SB3" "CleanRL" "TorchRL")
hps=("hps/SB3_ppo_defaultmujoco_unwrapped.yml" "hps/CleanRL_ppo_defaultmujoco_unwrapped.yml" "hps/TorchRL_ppo_defaultmujoco_unwrapped.yml" "hps/ppo.yml")
alg="ppo"
steps=3000005
rep=5
time=24:00:00
par=cpu-medium
mem=4G

for env in ${envs[@]};
do 
    for fw in ${fws[@]};
    do
        for hp in ${hps[@]};
        do
            bash job.sh \
                --fw=$fw \
                --alg=$alg \
                --env=$env \
                --steps=$steps \
                --rep=$rep \
                --time=$time \
                --par=$par \
                --mem=$mem \
                --hps=$hp
        done
    done
done