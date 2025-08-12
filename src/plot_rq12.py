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
    baselines = data.baselines 
    ylow = data.ylow
    yhigh = data.yhigh
    
    fig, axs = plt.subplots(nrows=8, ncols=4, figsize=(16, 28))

    for ax, col in zip(axs[0], hp_title):
        ax.set_title("$\mathcal{H}^{\\text{"+f'{col}'+"}}$")
    for ax in axs[-1]:
        ax.set_xlabel("Timesteps")
    for ax, row in zip(axs[:,0], ENVS):
        ax.set_ylabel(row, rotation=90, size='large')
        
    for i, hps in enumerate(HPS):
        for j, env in enumerate(ENVS):
            for k, p in enumerate([
                    "./"+base+f"/SB3/results/{data.fw}_{env}_{steps}/{hps}/",
                    "./"+base+f"/CleanRL/results/{data.fw}_{env}_{steps}/{hps}/",
                    "./"+base+f"/TorchRL/results/{data.fw}_{env}_{steps}/{hps}/"
                 ]):
                
                X,y,std = extractCurve(p,window=window,scalar=scalar)
                
                axs[j,i].plot(X, y, label=data.fw.upper()+"$^{\\text{"+fws[k]+"}}$", color=colors[k])
                #axs[j,i].set_xlabel("Timesteps")
                #axs[i].set_ylabel("Return")
                #axs[j,i].set_title(env)
                
                axs[j,i].fill_between(X, y-std, y+std, alpha=0.3, color=colors[k])
                
                if i == 3 and baselines[j] != None:
                    axs[j,i].axhline(y=baselines[j][1],linestyle='--',color='black')
                    #axs[j,i].axvline(x=1e6,linestyle='--',color='black')
                    axs[j,i].plot(baselines[j][0], baselines[j][1],color="black",marker=".", markersize=15)
                    
                    if hps == data.fw:
                        #print(X[(X > 9.8e5) & (X < 10.5e5)],y[(X > 9.8e5) & (X < 10.5e5)],env,p)
                        target = y[(X > baselines[j][0]-2e5) & (X < baselines[j][0]+5e5)][0]
                        std = std[(X > baselines[j][0]-2e5) & (X < baselines[j][0]+5e5)][0]
                        
                        m,l,u = getCI([(target,std)],confidence=95)
                        
                        if l[0] <= baselines[j][1] <= u[0]:
                            print("same\t", env, p)
                        elif baselines[j][1] < l[0]:
                            print("under\t", env, p)
                        elif baselines[j][1] > u[0]:
                            print("over\t", env, p)
                    
                axs[j,i].set_ylim(bottom=ylow[j]-0.1*ylow[j],top=yhigh[j]+0.1*yhigh[j]) # Add small margins such that the limits are not on the axis
            

                
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, fontsize='large')
    fig.subplots_adjust(hspace=0.5, bottom=0.05)
    fig.suptitle(f"Comparison of baseline framework {data.fw.upper()} implementations under various hyperparameter configurations",x=0.5,y=0.9)
    #plt.title(f'Reproduction of results PPO using baseline implementations of different frameworks')
    #plt.tight_layout()
    plt.savefig(f"./figs/reproduction-{data.fw}-{hps.split('_')[0]}-{'train' if not 'eval' in scalar else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    plt.clf()


    print("="*20)

ppo = Dataclass(
    "ppo",
    baselines = [(1e6,1800), (1e6,2230), (1e6,8000), (1e6,980), (1e6,-7), (1e6,108), (1e6,2950),None],   
    ylow = [0,0,0,0,-60,-25,0,0],
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
    ylow = [0,0,0,0,-20,-25,0,0],
    yhigh = [16000,4500,10000,1200,0,150,6500,8200]
)

#makePlot(ENVS=ENVS,HPS=HPS,scalar="rollout/ep_rew_mean",window=100)
makePlot(ppo, scalar="eval/mean_reward",window=50)
makePlot(td3, scalar="eval/mean_reward",window=50)
makePlot(sac, scalar="eval/mean_reward",window=50)
