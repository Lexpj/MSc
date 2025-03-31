import numpy as np
import matplotlib.pyplot as plt
import os

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join
from scipy.signal import savgol_filter
import matplotlib.patches as mpatches



def extractCurve(path, scalar="rollout/ep_rew_mean"):

    # Make collection arrays for each path and repetition
    KEYS = []
    VALUES = []

    # Check out what is included in the folder
    folders = [x[0]+"/" for x in os.walk(path)][1:]

    # If there are multiple experiments done, print a warning
    if folders[-1][-1] != 1:
        print("Multiple experiments done in this way!")

    # Extract the event loggers from each folder
    for rep in folders:
        log_file = [f for f in listdir(rep) if isfile(join(rep, f))]
        log_file = rep+[f for f in log_file if f[-3:] != ".pt"][0]
        # Read the logger file
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()
        print("Available scalar tags:", event_acc.Tags()['scalars'])  # List all scalar tags

        # Extract the given scalar info
        scalar_events = event_acc.Scalars(scalar)

        # Extract steps and values for plotting
        steps = np.array([e.step for e in scalar_events])
        values = np.array([e.value for e in scalar_events])
        
        # Save them for this experiment
        KEYS.append(steps)
        VALUES.append(values)

    minrun = min([len(x) for x in KEYS])
    KEYS = np.array([KEYS[i][:minrun] for i in range(len(KEYS))])
    VALUES = np.array([VALUES[i][:minrun] for i in range(len(VALUES))])

    return KEYS.mean(axis=0), VALUES.mean(axis=0), VALUES.std(axis=0)


def makePlot(paths):
    # Plot the values
    for p in paths:
        X,y,std = extractCurve(p)

        fw = p.split("/")[2]
        hp = p.split("/")[-2].split("_")[0]

        plt.plot(X, y,label=fw + " : " + hp)
        plt.fill_between(X, y-std, y+std ,alpha=0.3)
    
    
    plt.xlabel('Environment Steps')
    plt.ylabel('Mean episodic training reward (100)')
    plt.title(f'Comparison baseline frameworks using HPs from other baseline frameworks')
    plt.grid()
    plt.legend()

    plt.show()


include = [
    "./src/SB3/results/ppo_CartPole-v1_100000/CleanRL_ppo_defaultdiscrete/",
    "./src/SB3/results/ppo_CartPole-v1_100000/SB3_ppo_defaultdiscrete/",
    "./src/CleanRL/results/ppo_CartPole-v1_100000/CleanRL_ppo_defaultdiscrete/",
    "./src/CleanRL/results/ppo_CartPole-v1_100000/SB3_ppo_defaultdiscrete/"
]

makePlot(include)