#!/bin/bash

echo "Experiment with default parameters"
bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=CartPole-v1 \
    --steps=100000 \
    --rep=5 \
    --time=00:10:00 \
    --par=cpu-short \
    --mem=4G \
    --hps="./hps/CleanRL_ppo_defaultdiscrete.yml"

#echo "Experiment with high learning rate"
#bash job.sh \
#    --fw=SB3 \
#    --alg=ppo \
#    --env=CartPole-v1 \
#    --steps=100000 \
#    --rep=5 \
#    --time=00:10:00 \
#    --par=cpu-short \
#    --mem=4G \
#    --hps="./hps/SB3_ppo_morelearn.yml"
