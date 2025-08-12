import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join

import pickle
from plot_utils import *


def makePlot(hps, scalar="rollout/ep_rew_mean",window=50,ylow=[],yhigh=[]):
    # Plot the values
    base = "src/"
    base = ""
    steps = 3000010
    
    colors = ['blue','orange','green']
    fws = ["SB3", "CleanRL","TorchRL"]
    envs = ['HalfCheetah-v4', "Walker2d-v4", "Hopper-v4", "Ant-v4"]
    envs = ["HalfCheetah-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4","Ant-v4"]
    algs = ["PPO", "SAC", "TD3"]
    fig, axs = plt.subplots(nrows=len(envs), ncols=3,figsize=(16, 28))

    for ax, col in zip(axs[0], fws):
        ax.set_title(f"FW: {col}")
    for ax in axs[-1]:
        ax.set_xlabel("Timesteps")
    for ax, row in zip(axs[:,0], envs):
        ax.set_ylabel(row, rotation=90, size='large')
     
     
    for i, fw in enumerate(fws):
        for j, env in enumerate(envs):
            if hps == "indiv":
                includes = [
                    "./"+base+f"/{fw}/results/ppo_{env}_{steps-5}/{fw}_ppo_defaultmujoco_unwrapped/",
                    "./"+base+f"/{fw}/results/sac_{env}_{steps}/{fw}_sac_defaultmujoco/",
                    "./"+base+f"/{fw}/results/td3_{env}_{steps}/{fw}_td3_defaultmujoco/",
                ]
            elif hps == None:
                includes = [
                    "./"+base+f"/{fw}/results/ppo_{env}_{steps-5}/ppo/",
                    "./"+base+f"/{fw}/results/sac_{env}_{steps}/sac/",
                    "./"+base+f"/{fw}/results/td3_{env}_{steps}/td3/",
                ]
            else:
                includes = [
                    "./"+base+f"/{fw}/results/ppo_{env}_{steps-5}/{hps}_ppo_defaultmujoco_unwrapped/",
                    "./"+base+f"/{fw}/results/sac_{env}_{steps}/{hps}_sac_defaultmujoco/",
                    "./"+base+f"/{fw}/results/td3_{env}_{steps}/{hps}_td3_defaultmujoco/",
                ]
                 
            for k, p in enumerate(includes):
                try:
                    X,y,std = extractCurve(p,window=window,scalar=scalar)
                    print(len(X))
                    axs[j,i].plot(X, y, label=algs[k], color=colors[k])
                #axs[j,i].set_xlabel("Timesteps")
                #axs[i].set_ylabel("Return")
                #axs[j,i].set_title(env)
                
                    axs[j,i].fill_between(X, y-std, y+std, alpha=0.3, color=colors[k])
                

                except:pass
                axs[j,i].set_ylim(bottom=ylow[j]-0.1*ylow[j],top=yhigh[j]+0.1*yhigh[j]) # Add small margins such that the limits are not on the axis
            

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='large')
    fig.subplots_adjust(hspace=0.5, bottom=0.05)
    
    #plt.title(f'Reproduction of results PPO using baseline implementations of different frameworks')
    #plt.tight_layout()
    fig.suptitle("Comparison ranking of algorithms of frameworks under hyperparameter configuration "+"$\mathcal{H}^{\\text{"+f'{hps if hps != None else "OG"}'+"}}$",x=0.5,y=0.9)
    plt.savefig(f"./figs/rq3-{hps}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    plt.clf()




HPS = ["SB3","CleanRL","TorchRL",None]

#makePlot(HPS=HPS,scalar="rollout/ep_rew_mean",window=100)
makePlot(hps="SB3",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="CleanRL",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="TorchRL",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps=None,scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="indiv",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
