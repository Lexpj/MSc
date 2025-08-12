#!/bin/bash

env=CartPole-v1
steps=1000000
rep=5
time=01:00:00
par=cpu-short
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
    --hps="./hps/SB3_ppo_defaultdiscrete.yml"

bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/CleanRL_ppo_defaultdiscrete.yml"

bash job.sh \
    --fw=SB3 \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/CleanRL_ppo_defaultdiscrete.yml"

bash job.sh \
    --fw=CleanRL \
    --alg=ppo \
    --env=$env \
    --steps=$steps \
    --rep=$rep \
    --time=$time \
    --par=$par \
    --mem=$mem \
    --hps="./hps/SB3_ppo_defaultdiscrete.yml"
    

