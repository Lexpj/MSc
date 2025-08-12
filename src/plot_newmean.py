import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join



def mean_std(d, window):
    return subsample(d, window)
    return savgol(d, window)

def savgol(d, window):
    y = np.array([np.mean(x[1]) for x in sorted(list(d.items()))])
    std = np.array([np.std(x[1]) for x in sorted(list(d.items()))])
    X = np.array(sorted(list(d.keys())))

    if window != None:        
        y = savgol_filter(y, window, 1)

    return X, y, std
    
def subsample(d, window):
    y = np.array([np.mean(x[1]) for x in sorted(list(d.items()))])
    std = np.array([np.std(x[1]) for x in sorted(list(d.items()))])
    X = np.array(sorted(list(d.keys())))

    
    if window == None:
        return X, y, std
    
    # Windowsize must be based on the amount of reported metrics and the many points of the window
    windowsize = len(X)//window
    
    X = X[0:len(X):windowsize]
    y = y[0:len(y):windowsize]
    std = std[0:len(std):windowsize]
    
    assert len(y) == len(std) == len(X), f"Something is fishy: X:{X}, y:{y}, std:{std}"
    
    return X, y, std

def extractCurve(path, scalar, window):

    # Make collection arrays for each path and repetition
    KV = {}

    # Check out what is included in the folder
    folders = [x[0]+"/" for x in os.walk(path)][1:]
    single_runs = []

    # Extract the event loggers from each folder
    for rep in folders:
        log_file = [f for f in listdir(rep) if isfile(join(rep, f))]
        log_file = rep+[f for f in log_file if f[-3:] != ".pt"][0]
        single_runs.append({})
        # Read the logger file
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()
        #print("Available scalar tags:", event_acc.Tags()['scalars'])  # List all scalar tags

        # Extract the given scalar info
        try:
            scalar_events = event_acc.Scalars(scalar)
        except:
            scalar_events = event_acc.Scalars("eval/reward")
        # Extract steps and values for plotting
        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])
        single_runs[-1]["steps"] = steps
        single_runs[-1]["values"] = values
        
        # Save them for this experiment
        for key,value in zip(steps,values):
            KV[key] = KV.get(key,[]) + [value]
    #print(KV)
    step,avg,std = mean_std(KV,window=window)

    return step, avg, std, single_runs


def makePlot(paths, window=None, scalar="rollout/ep_rew_mean",exp_name="",evaluate=False,single_runs=False):
    # Plot the values
    colors = ["blue","orange","green","red"]
    linesstyles=["-","--"]
    for i,p in enumerate(paths):
        print(f"Extracting curve {scalar} of {exp_name}, path={p}")
        X,y,std,sr = extractCurve(p,window=window,scalar=scalar)
        print(len(X))
        fw = p.split("/")[2]
        hp = p.split("/")[-2].split("_")[0]

        plt.plot(X, y,label="SAC$^{"+fw+"}(\mathcal{H}^{\\text{"+ hp + "}})$",color=colors[i%4],linestyle=linesstyles[i//4])
        plt.fill_between(X, y-std, y+std ,alpha=0.1,color=colors[i%4])
        
        #if single_runs:
        #    for run in sr:
        #        plt.plot(run["steps"],run["values"],color=colors[i],linestyle="--",alpha=0.1)
    
    
    plt.xlabel('Environment Steps')
    if evaluate:
        plt.ylabel('Mean evaluation return')
    else:
        plt.ylabel('Mean training return')
    #plt.title(f'Comparing HPs of SB3 and CleanRL on {exp_name}')
    plt.grid()
    plt.legend()
    #plt.ylim((0,17500))
    #plt.xlim((0,1e6))
    #plt.savefig(f"./figs/{exp_name}-{'train' if not evaluate else 'eval'}{'-w'+str(window) if window != None else ''}.png")
    plt.show()
    plt.clf()

def CartPoleWrapped():
    env = "CartPole-v1"
    steps = 1000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultdiscretewrapped/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultdiscretewrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultdiscretewrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultdiscretewrapped/"
    ]
def CartPoleUnwrapped():
    env = "CartPole-v1"
    steps = 1000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultdiscrete/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultdiscrete/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultdiscrete/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultdiscrete/"
    ]
def HalfCheetahWrapped():
    env = "HalfCheetah-v4"
    steps = 3000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_wrapped/"
    ]
def HalfCheetahUnwrapped():
    env = "HalfCheetah-v4"
    steps = 3000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_unwrapped/"
    ]
def AntWrapped():
    env = "Ant-v4"
    steps = 3000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_wrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_wrapped/"
    ]
def AntUnwrapped():
    env = "Ant-v4"
    steps = 3000005
    return ["./"+base+f"/SB3/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/SB3/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/CleanRL_ppo_defaultmujoco_unwrapped/",
    "./"+base+f"/CleanRL/results/ppo_{env}_{steps}/SB3_ppo_defaultmujoco_unwrapped/"
    ]
        
def SacHalfCheetah():
    env = "HalfCheetah-v4"
    steps = 3000008
    return ["./"+base+f"/SB3/results/sac_{env}_{steps}/CleanRL_sac_defaultmujoco/",
    "./"+base+f"/SB3/results/sac_{env}_{steps}/SB3_sac_defaultmujoco/",
    "./"+base+f"/CleanRL/results/sac_{env}_{steps}/CleanRL_sac_defaultmujoco/",
    "./"+base+f"/CleanRL/results/sac_{env}_{steps}/SB3_sac_defaultmujoco/"
    ]
    
def Td3HalfCheetah():
    env = "HalfCheetah-v4"
    steps = 3000005
    return ["./"+base+f"/SB3/results/td3_{env}_{steps}/CleanRL_td3_defaultmujoco/",
    "./"+base+f"/SB3/results/td3_{env}_{steps}/SB3_td3_defaultmujoco/",
    "./"+base+f"/CleanRL/results/td3_{env}_{steps}/CleanRL_td3_defaultmujoco/",
    "./"+base+f"/CleanRL/results/td3_{env}_{steps}/SB3_td3_defaultmujoco/"
    ]
    

def Test():
    env = "HalfCheetah-v4"
    alg = "sac"
    steps = 3000010
    steps = 1000008
    return ["./"+base+f"/SB3/results/{alg}_{env}_{steps}/CleanRL_{alg}_defaultmujoco/",
    "./"+base+f"/SB3/results/{alg}_{env}_{steps}/SB3_{alg}_defaultmujoco/",
    "./"+base+f"/CleanRL/results/{alg}_{env}_{steps}/CleanRL_{alg}_defaultmujoco/",
    "./"+base+f"/CleanRL/results/{alg}_{env}_{steps}/SB3_{alg}_defaultmujoco/",
    ]
def Test2():
    env = "HalfCheetah-v4"
    alg = "sac"
    steps = 1000008
    return ["./"+base+f"/SB3/results/{alg}_{env}_{steps}/CleanRL_{alg}_defaultmujoco_envs/",
    "./"+base+f"/SB3/results/{alg}_{env}_{steps}/SB3_{alg}_defaultmujoco_envs/",
    "./"+base+f"/CleanRL/results/{alg}_{env}_{steps}/CleanRL_{alg}_defaultmujoco_envs/",
    "./"+base+f"/CleanRL/results/{alg}_{env}_{steps}/SB3_{alg}_defaultmujoco_envs/"
    ]

def HalfCheetahTorchRL():
    env = "HalfCheetah-v4"
    steps = 3000008
    return HalfCheetahUnwrapped() + [
    "./"+base+f"/TorchRL/results/ppo_{env}_{steps}/TorchRL_ppo_defaultmujoco_unwrapped/"
    
    ]  
def HalfCheetahCleanRLTorchRL():
    env = "HalfCheetah-v4"
    steps = 1000008
    return HalfCheetahUnwrapped() + [
    "./"+base+f"/TorchRL/ppo/CleanRL/"
    ] 
    
def hpGen(lst):
    """
    In the form [alg,fw,hp,env]
    general use cases only
    """
    hps = []
    for alg,fw,hp,env in lst:
        hps.append("./"+base+f"/{fw}/results/{alg}_{env}_{3000010 if alg != 'ppo' else 3000005}/{hp}_{alg}_defaultmujoco{'_unwrapped' if alg == 'ppo' else ''}/")
    return hps    


       
base = "src/"
base = ""

includes = [Test()]#,Test2(),SacHalfCheetah(),]
exps = ["Ant-v4, $n_{envs}=1$", "TD3 HalfCheetah-v4, $n_{envs}=8$"]


includes = [HalfCheetahTorchRL()]#, HalfCheetahCleanRLTorchRL()]
exps = ["HalfCheetah", "TorchRL PPO with HPS:CleanRL"]

includes = [[
    "./"+base+f"/TorchRL/results/sac_Swimmer-v4_3000010/SB3_sac_defaultmujoco/",
    "./"+base+f"/TorchRL/results/sac_Swimmer-v4_3000010/CleanRL_sac_defaultmujoco/",
    "./"+base+f"/TorchRL/results/sac_Swimmer-v4_3000010/TorchRL_sac_defaultmujoco/",
    ]]
#includes = [Test(), Test2()]
#exps = ["SAC HalfCheetah-v4, $n_{envs}=1$", "SAC HalfCheetah-v4, $n_{envs}=8$"]


exps = ["PPO CartPole-v1", "PPO CartPole-v1 (wrapped)", "PPO HalfCheetah-v4", "PPO HalfCheetah-v4 (wrapped)", "PPO Ant-v4", "PPO Ant-v4 (wrapped)"]
includes = [CartPoleUnwrapped(), CartPoleWrapped(), HalfCheetahUnwrapped(), HalfCheetahWrapped(), AntUnwrapped(), AntWrapped()]
includes = [Test(),Test2()]
exps = ["SAC HalfCheetah-v4, $n_{envs}=1$", "SAC HalfCheetah-v4, $n_{envs}=8$"]

includes = [hpGen([
    ['sac',i,'SB3','HalfCheetah-v4'] for i in ['SB3','CleanRL','TorchRL']
])]
print(includes)

exps = ["YES"]

for include, exp in zip(includes,exps):
    #makePlot(include, window=10, scalar="rollout/ep_rew_mean",exp_name=exp)
    #makePlot(include, window=10, scalar="eval/mean_reward",exp_name=exp,evaluate=True)
    #makePlot(include, scalar="rollout/ep_rew_mean",exp_name=exp,window=100)
    makePlot(include, scalar="eval/mean_reward",exp_name=exp,evaluate=True,window=50)
#makePlot(include, [7,231])
