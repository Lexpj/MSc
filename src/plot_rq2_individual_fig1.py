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
    fig, ax = plt.subplots(1, 1, 
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
        
        
        color = "blue"
        if fw == "CleanRL" or alg == "td3":
            color = "orange"
        if fw == "TorchRL" or alg == "sac":
            color = "green"
        line, = ax.plot(X, y, label=fws[i], color=color, linewidth=2)
        ax.fill_between(X, y - std, y + std, alpha=0.1, color=color)

        
        legend_handles.append(line)
        legend_labels.append(fw)


    ax.set_xlabel('Environment Steps')
    if evaluate:
        ax.set_ylabel('Mean evaluation return')
    else:
        ax.set_ylabel('Mean training return')
    ax.set_title(env + f" with {alg.upper()}"+"$\mathcal{H}^\\text{" +hp+ "}$" )
    ax.set_ylim((0,6000))
    
    #ax.axhline(y=6000,linestyle='--',color='black')
                  #axs[j,i].axvline(x=1e6,linestyle='--',color='black')
    #ax.plot(3e6, 6000, color="black",marker=".", markersize=15)
                    

    fig.legend(legend_handles, legend_labels, loc='lower center', ncol=len(paths), bbox_to_anchor=(0.5, 0.00))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig(f"./figs1/{alg}-{hp}-{env}-{fw}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    plt.clf()
    

def full(env,hps,alg,fw):
    steps = 3000005 if alg == "ppo" else 3000010
    if hps != alg:
        hpsstr = f"{hps}_{alg}_defaultmujoco{'_unwrapped' if alg == 'ppo' else ''}"
    else:
        hpsstr = f"{hps}"
    return ["./"+base+f"/{fw}/results/{alg}_{env}_{steps}/{hpsstr}/"]
    
    
base = ""

includes = []
for hp in ["SB3", "CleanRL", "TorchRL", None]:
    for alg in ["ppo", "td3", "sac"]:
        for fw in ["SB3", "CleanRL", "TorchRL"]:
            _hp = hp if hp != None else alg
            includes.append(full("Walker2d-v4", _hp, alg, fw))
            
includes = [full("Walker2d-v4", "SB3", "sac", "TorchRL")]


for include in includes:
    #makePlot(include, window=10, scalar="rollout/ep_rew_mean")
    #makePlot(include, window=10, scalar="eval/mean_reward",evaluate=True)
    #makePlot(include, scalar="rollout/ep_rew_mean",window=100)
    makePlot(include, scalar="eval/mean_reward",evaluate=True,window=50)
#makePlot(include, [7,231])
