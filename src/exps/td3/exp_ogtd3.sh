#!/bin/bash

envs=("InvertedDoublePendulum-v4" "InvertedPendulum-v4" "Reacher-v4" "Swimmer-v4" "Walker2d-v4" "HalfCheetah-v4" "Hopper-v4" "Ant-v4")
fws=("SB3" "CleanRL" "TorchRL")
hps=("hps/SB3_td3_defaultmujoco.yml" "hps/CleanRL_td3_defaultmujoco.yml" "hps/TorchRL_td3_defaultmujoco.yml" "hps/td3.yml")
alg="td3"
steps=3000010
rep=5
time=48:00:00
par=cpu-long
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