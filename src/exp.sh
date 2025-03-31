#!/bin/bash

echo "Experiment SB3  with default discrete parameters CleanRL"

#bash job.sh \
#    --fw=SB3 \
#    --alg=ppo \
#    --env=CartPole-v1 \
#    --steps=100000 \
#    --rep=3 \
#    --time=00:20:00 \
#    --par=cpu-short \
#    --mem=4G \
#    --hps="./hps/SB3_ppo_defaultdiscrete.yml"

echo "Experiment CleanRL  with default discrete parameters SB3"
bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=CartPole-v1 \
    --steps=100000 \
    --rep=3 \
    --time=00:20:00 \
    --par=cpu-short \
    --mem=4G \
    --hps="./hps/CleanRL_ppo_defaultdiscrete.yml"

#bash job.sh \
#    --fw=SB3 \
#    --alg=ppo \
#    --env=CartPole-v1 \
#    --steps=100000 \
#    --rep=3 \
#    --time=00:20:00 \
#    --par=cpu-short \
#    --mem=4G \
#    --hps="./hps/CleanRL_ppo_defaultdiscrete.yml"

echo "Experiment CleanRL  with default discrete parameters SB3"
bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=CartPole-v1 \
    --steps=100000 \
    --rep=3 \
    --time=00:20:00 \
    --par=cpu-short \
    --mem=4G \
    --hps="./hps/SB3_ppo_defaultdiscrete.yml"
    
#echo "Experiment with default atari parameters CleanRL"
#bash job.sh \
#    --fw=CleanRL \
#    --alg=ppo \
#    --env=ALE/Pong-v4 \
#    --steps=100000 \
#    --rep=3 \
#    --time=00:10:00 \
#    --par=cpu-short \
#    --mem=4G \

    
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
