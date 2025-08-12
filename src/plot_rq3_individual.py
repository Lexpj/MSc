import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join

import pickle
from plot_utils import *


def makePlot(hps, env, scalar="rollout/ep_rew_mean",window=50,ylow=[],yhigh=[]):
    # Plot the values
    base = "src/"
    base = ""
    steps = 3000010
    
    colors = ['blue','orange','green']
    fws = ["SB3", "CleanRL","TorchRL"]
    envs = ['HalfCheetah-v4', "Walker2d-v4", "Hopper-v4", "Ant-v4"]
    envs = ["HalfCheetah-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4","Ant-v4"]
    ylow = [ylow[envs.index(env)]]
    yhigh = [yhigh[envs.index(env)]]
    envs = [env]
    
    algs = ["PPO", "SAC", "TD3"]
    fig, axs = plt.subplots(nrows=2, ncols=3,figsize=(16, 5),gridspec_kw={'height_ratios': [4, 1]}, )

    for ax, col in zip(axs[0], fws):
        ax.set_title(f"Framework: {col}")
    for ax in axs[-1]:
        ax.set_xlabel("Timesteps")
    #for ax, row in zip(axs[:,0], envs):
    #    ax.set_ylabel(row, rotation=90, size='large')
     
     
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
                    axs[0,i].plot(X, y, label=algs[k], color=colors[k])
                #axs[j,i].set_xlabel("Timesteps")
                #axs[i].set_ylabel("Return")
                #axs[j,i].set_title(env)
                
                    axs[0,i].fill_between(X, y-std, y+std, alpha=0.3, color=colors[k])
                
                    m,l,u = getCI([(y[-1],std[-1])])
                            
                    center = 0 

                    axs[1,i].barh(k, width=u[0] - l[0], left=l[0], height=0.12, color=colors[k], alpha=0.6, edgecolor='black')

                    # Plot the mean as a vertical line in the center of the bar
                    axs[1,i].plot([m[0], m[0]], [k - 0.06, k + 0.06], color='black', linewidth=2)

                    axs[1,i].plot(m[0], k, 'o', color=colors[k], markersize=10)

                    axs[1,i].set_ylabel("")
                    axs[1,i].set_yticks([])
                    axs[1,i].set_xlim(left=ylow[j]-0.1*ylow[j],right=yhigh[j]+0.1*yhigh[j])
                except Exception as e:
                    print(e)
                axs[0,i].set_ylim(bottom=ylow[j]-0.1*ylow[j],top=yhigh[j]+0.1*yhigh[j]) # Add small margins such that the limits are not on the axis
            
                    
    handles, labels = axs[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='large')
    fig.subplots_adjust(hspace=0.5, bottom=0.05)
    
    #plt.title(f'Reproduction of results PPO using baseline implementations of different frameworks')
    #plt.tight_layout()
    fig.suptitle("Comparison ranking of algorithms of frameworks under hyperparameter configuration "+"$\mathcal{H}^{\\text{"+f'{hps if hps != None else "OG"}'+"}}$",x=0.5,y=1)
    #fig.tight_layout()
    fig.subplots_adjust(
    top=0.9,    # Leave space at the top for the title
    bottom=0.15,  # Optional: avoid cutting off x-axis labels
    hspace=0.3,  # Reduce vertical spacing between rows
    #wspace=0.3   # Optional: adjust horizontal spacing
    )
    #plt.show()
    plt.savefig(f"./figs/rq3-{hps}-{envs[0]}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    
    plt.clf()




HPS = ["SB3","CleanRL","TorchRL",None]

#makePlot(HPS=HPS,scalar="rollout/ep_rew_mean",window=100)
makePlot(hps="CleanRL",env="Walker2d-v4",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="SB3",env="Swimmer-v4",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="indiv",env="Walker2d-v4",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
makePlot(hps="SB3",env="Walker2d-v4",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
#makePlot(hps="CleanRL",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
#makePlot(hps="TorchRL",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
#makePlot(hps=None,scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
#makePlot(hps="indiv",scalar="eval/mean_reward",ylow=[0,0,0,0,-30,0,0,0],yhigh=[15000,4000,11000,1250,0,150,6000,8000])
