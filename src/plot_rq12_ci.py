import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join

import pickle
from plot_utils import *

class Dataclass:
    def __init__(self, fw, baselines, ylow, yhigh):
        self.fw = fw
        self.baselines = baselines
        self.ylow = ylow
        self.yhigh = yhigh

def makePlot(data,scalar="rollout/ep_rew_mean",window=None):
    # Plot the values
    base = "src/"
    base = ""
    steps = 3000010 if data.fw != 'ppo' else 3000005
    
    ENVS = ["HalfCheetah-v4", "Hopper-v4", "InvertedDoublePendulum-v4", "InvertedPendulum-v4", "Reacher-v4", "Swimmer-v4", "Walker2d-v4","Ant-v4"]
    HPS = [f"SB3_{data.fw}_defaultmujoco{'_unwrapped' if data.fw == 'ppo' else ''}", f"CleanRL_{data.fw}_defaultmujoco{'_unwrapped' if data.fw == 'ppo' else ''}", f"TorchRL_{data.fw}_defaultmujoco{'_unwrapped' if data.fw == 'ppo' else ''}", data.fw]
    
    colors = ['blue','orange',"green"]
    hp_title = ["SB3", "CleanRL", "TorchRL", data.fw]
    fws = ['SB3', 'CleanRL',"TorchRL"]
    
    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 14))

    for ax, col in zip(axs[0], hp_title):
        ax.set_title("$\mathcal{H}^{\\text{"+f'{col}'+"}}$")
    for ax in axs[-1]:
        ax.set_xlabel("Timesteps")
    for ax, row in zip(axs[:,0], ENVS):
        ax.set_ylabel(row, rotation=90, size='small')
        
        
    for i, hps in enumerate(HPS):
        for j, env in enumerate(ENVS):
            lst = []
            for k, p in enumerate([
                    "./"+base+f"/SB3/results/{data.fw}_{env}_{steps}/{hps}/",
                    "./"+base+f"/CleanRL/results/{data.fw}_{env}_{steps}/{hps}/",
                    "./"+base+f"/TorchRL/results/{data.fw}_{env}_{steps}/{hps}/"
                 ]):
                
                X,y,std = extractCurve(p,window=window,scalar=scalar)
                lst.append((y[-1],std[-1]))
            
            m,l,u = getCI(lst)
                
            for k in range(3):
                # Use a small vertical offset to separate the bars visually
                center = 0  # All bars centered at y=0

                # Plot the confidence interval as a horizontal bar
                axs[j,i].barh(k, width=u[k] - l[k], left=l[k], height=0.12, color=colors[k], alpha=0.6, edgecolor='black', label=f'{fws[k]}')
                if hps == "ppo" and env == "InvertedDoublePendulum-v4":
                    X,y,std = extractCurve("./"+base+f"/CleanRL/results/{data.fw}_{env}_{steps}/{hps}/",window=window,scalar=scalar)
                    print(y[-1],std[-1],m,l,u)
                # Plot the mean as a vertical line in the center of the bar
                axs[j,i].plot([m[k], m[k]], [k - 0.06, k + 0.06], color='black', linewidth=2)

                axs[j,i].plot(m[k], k, 'o', color=colors[k], markersize=10)
                if i != 0:
                    axs[j,i].set_ylabel("")
                axs[j,i].set_yticks([])
                
            axs[j,i].set_xlim(left=data.ylow[j]-0.1*data.ylow[j],right=data.yhigh[j]+0.1*data.yhigh[j]) # Add small margins such that the limits are not on the axis

  
                
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='large')
    fig.subplots_adjust(hspace=0.5, bottom=0.05)
    fig.suptitle(f"Comparison of baseline framework {data.fw.upper()} implementations under various hyperparameter configurations",x=0.5,y=0.95)
    #plt.title(f'Reproduction of results PPO using baseline implementations of different frameworks')
    #plt.tight_layout()
    plt.savefig(f"./figs/reproduction-{data.fw}-{hps.split('_')[0]}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}-ci.png")
    plt.clf()




ppo = Dataclass(
    "ppo",
    baselines = [(1e6,1800), (1e6,2230), (1e6,8000), (1e6,980), (1e6,-7), (1e6,108), (1e6,2950),None],   
    ylow = [0,0,0,0,-10,-25,0,0],
    yhigh = [5000,4000,10000,1000,0,125,5000,5000] 
)
td3 = Dataclass(
    "td3",
    baselines = [(1e6,9500), (1e6,3250), (1e6,9000), (1e6,980), (1e6,-4), None, (1e6,4500), (1e6,4100)],
    ylow = [0,-200,0,0,-10,-25,0,0],
    yhigh = [15000,4000,11000,1250,0,175,6500,7000]
)
sac = Dataclass(
    "sac",
    baselines = [(3e6,15200), (1e6,3300), None, None, None, None, (3e6,6000), (3e6,5500)], # Humanoid = 8000 
    ylow = [0,0,0,0,-10,-25,0,0],
    yhigh = [16000,4500,10000,1200,0,150,6500,8200]
)

#makePlot(ENVS=ENVS,HPS=HPS,scalar="rollout/ep_rew_mean",window=100)
makePlot(ppo, scalar="eval/mean_reward",window=None)
makePlot(td3, scalar="eval/mean_reward",window=None)
makePlot(sac, scalar="eval/mean_reward",window=None)
