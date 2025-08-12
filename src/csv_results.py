import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join


import pickle
from plot_utils import *


def resCSV(alg, scalar="eval/mean_reward",window=None):
    # Plot the values
    base = "src/"
    base = ""
    steps = 3000005
    
    hp_title = ["SB3", "CleanRL", "TorchRL", alg]
    fws = ['SB3', 'CleanRL', "TorchRL"]
    envs = ["HalfCheetah-v4", "Ant-v4", "Hopper-v4", "Walker2d-v4", "InvertedPendulum-v4", "InvertedDoublePendulum-v4", "Swimmer-v4", "Reacher-v4"]
       
    res = {hp:{fw:{} for fw in fws} for hp in hp_title}
    
    for i, hps in enumerate(hp_title):
        for j, fw in enumerate(fws):
            for k, env in enumerate(envs):
                if hps != alg:
                    p = "./"+base+f"/{fw}/results/{alg}_{env}_{steps + 5 if alg != 'ppo' else steps}/{hps}_{alg}_defaultmujoco{'_unwrapped' if alg == 'ppo' else ''}/"
                else:
                    p = "./"+base+f"/{fw}/results/{alg}_{env}_{steps + 5 if alg != 'ppo' else steps}/{hps}/"
                X,y,std = extractCurve(p,window=window,scalar=scalar)  
                try:
                    
                    m,l,u = getCI([(y[-1],std[-1])],confidence=95)
                    if hps == "ppo" and fw == "CleanRL":
                        print(y[-1],std[-1],m,l,u)
                    res[hps][fw][env] = f"{round(y[-1],1)}$\pm${round(u[0]-m[0],1)}"
                except:
                    res[hps][fw][env] = "-"

    return res

def convertToLaTeX(res,alg):
    print(res)
    
    
    hp_title = ["SB3", "CleanRL", "TorchRL", alg]
    fws = ['SB3', 'CleanRL', "TorchRL"]
    envs = ["HalfCheetah-v4", "Ant-v4", "Hopper-v4", "Walker2d-v4", "InvertedPendulum-v4", "InvertedDoublePendulum-v4", "Swimmer-v4", "Reacher-v4"]
    s = "\\hline HPS & FW "
    for e in envs[0:4]:
        s += "& " + e  
    s += "\\\\"
    
    
    
    for hp in res.keys():
        s += "\hline "
        for fw in res[hp].keys():
            s += f"{hp} & {fw} " 
            for env in envs[0:4]:
                s += "& " + res[hp][fw][env]
            s += "\\\\"
    s += "\\hline\\hline "
    s += "HPS & FW "
    for e in envs[4:]:
        if e == "InvertedDoublePendulum-v4":
            e = "Inv.Doub.Pend.-v4"
        elif e == "InvertedPendulum-v4":
            e = "Inv.Pend.-v4"
        s += "& " + e  
    s += "\\\\"
    
    
    
    for hp in res.keys():
        s += "\hline "
        for fw in res[hp].keys():
            s += f"{hp} & {fw} " 
            for env in envs[4:]:
                s += "& " + res[hp][fw][env]
            s += "\\\\"    
    s += "\\hline"
    print(s) 
                
    
    


ALG = "ppo"
convertToLaTeX(resCSV(ALG),ALG)
