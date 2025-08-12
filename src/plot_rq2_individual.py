import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join

from plot_utils import *




def makePlot(paths, window=None, scalar="rollout/ep_rew_mean", evaluate=False):
    # Prepare 3 subplots side by side
    fig, ax = plt.subplots(2, 1, 
                               gridspec_kw={'height_ratios': [4, 1]}, 
                               figsize=(8, 6))
    colors = ["blue", "orange", "green", "red"]
    fws = hps = ["SB3", "CleanRL", "TorchRL"]

    linesstyles = ["-", "--"]

    legend_handles = []
    legend_labels = []

    for i, p in enumerate(paths):
        print(f"Extracting curve {scalar} of path={p}")
        X, y, std = extractCurve(p, window=window, scalar=scalar)
        print(len(X))
        fw = p.split("/")[2]
        hp = p.split("/")[-2].split("_")[0]
        env = p.split("/")[-3].split("_")[1]
        alg = p.split("/")[-3].split("_")[0]
        
        line, = ax[0].plot(X, y, label=fws[i], color=colors[i], linewidth=1)
        ax[0].fill_between(X, y - std, y + std, alpha=0.1, color=colors[i])

        
        legend_handles.append(line)
        legend_labels.append(fw)


        m,l,u = getCI([(y[-1],std[-1])])
                
        center = 0 

        ax[1].barh(i, width=u[0] - l[0], left=l[0], height=0.12, color=colors[i], alpha=0.6, edgecolor='black')

        # Plot the mean as a vertical line in the center of the bar
        ax[1].plot([m[0], m[0]], [i - 0.06, i + 0.06], color='black', linewidth=2)

        ax[1].plot(m[0], i, 'o', color=colors[i], markersize=10)

        ax[1].set_ylabel("")
        ax[1].set_yticks([])
        

    ax[0].set_xlabel('Environment Steps')
    if evaluate:
        ax[0].set_ylabel('Mean evaluation return')
    else:
        ax[0].set_ylabel('Mean training return')
    ax[0].set_title(env + f" with {alg.upper()}"+"$\mathcal{H}^\\text{" +hp+ "}$" )
    

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(paths), bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"./figs/{alg}-{hp}-{env}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    plt.clf()
    

def full(env,hps,alg):
    steps = 3000005 if alg == "ppo" else 3000010
    if hps != alg:
        hpsstr = f"{hps}_{alg}_defaultmujoco{'_unwrapped' if alg == 'ppo' else ''}"
    else:
        hpsstr = f"{hps}"
    return ["./"+base+f"/SB3/results/{alg}_{env}_{steps}/{hpsstr}/",
    "./"+base+f"/CleanRL/results/{alg}_{env}_{steps}/{hpsstr}/",
    "./"+base+f"/TorchRL/results/{alg}_{env}_{steps}/{hpsstr}/"]
    
    
base = ""

includes = [
    full("HalfCheetah-v4","CleanRL","ppo"),
    full("Hopper-v4","CleanRL","ppo"),
    full("InvertedDoublePendulum-v4","TorchRL","ppo"),
    full("Swimmer-v4","SB3","ppo"),
    full("Swimmer-v4","ppo","ppo"),
    full("Walker2d-v4","SB3","ppo"),
    
    full("HalfCheetah-v4","SB3", "td3"),
    full("Hopper-v4","SB3", "td3"),
    full("Walker2d-v4","SB3", "td3"),
    full("InvertedDoublePendulum-v4","CleanRL", "td3"),
    full("Reacher-v4","td3", "td3"),
    full("Walker2d-v4","TorchRL", "td3"),
    
    full("HalfCheetah-v4","SB3", "sac"),
    full("HalfCheetah-v4","TorchRL", "sac"),
    full("Swimmer-v4","CleanRL", "sac"),
    full("Swimmer-v4","sac", "sac"),
    full("Ant-v4","CleanRL", "sac"),
    full("Ant-v4","sac", "sac"),
]

for include in includes:
    #makePlot(include, window=10, scalar="rollout/ep_rew_mean")
    #makePlot(include, window=10, scalar="eval/mean_reward",evaluate=True)
    #makePlot(include, scalar="rollout/ep_rew_mean",window=100)
    makePlot(include, scalar="eval/mean_reward",evaluate=True,window=50)
#makePlot(include, [7,231])
