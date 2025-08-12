import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join

def extractCurve(path, scalar, window, total_steps):
    # Check out what is included in the folder
    folders = [x[0]+"/" for x in os.walk(path)][1:]
    single_runs = {}

    # Extract the event loggers from each folder
    for rep in folders:
        log_file = [f for f in listdir(rep) if isfile(join(rep, f))]
        log_file = rep+[f for f in log_file if f[-3:] != ".pt"][0]
        
        # Read the logger file
        event_acc = EventAccumulator(log_file)
        event_acc.Reload()
        #print("Available scalar tags:", event_acc.Tags()['scalars'])  # List all scalar tags

        # Extract the given scalar info
        try:
            scalar_events = event_acc.Scalars(scalar)
        except:
            continue
        # Extract steps and values for plotting
        steps = np.array([e.step for e in scalar_events])

        # If the last step does not differ with more than 10000, we know the run has finished
        single_runs[rep] = abs(max(steps) - total_steps) <= 15000
        if abs(max(steps) - total_steps) > 15000: print(abs(max(steps) - total_steps))

    return single_runs



def getExperiments(folder,alg):
    experiments = [f for f in os.listdir("./"+folder+"/results") if os.path.isdir(os.path.join(folder,"results",f))]
    D = {experiment:{} for experiment in experiments if alg in experiment}
    print(folder)
    for key in D.keys():
        hps = [f for f in os.listdir("./"+folder+"/results/"+key) if os.path.isdir(os.path.join(folder,"results",key,f))]
        D[key] = {h: {} for h in hps}
        f2_displayed = False
        for key_hps in D[key].keys():
            f = "./"+folder+"/results/"+key+"/"+key_hps+"/"
            # Extract as normal
            try:
                truths = extractCurve(f,window=None,scalar="eval/mean_reward",total_steps=int(key.split("_")[-1]))
            except Exception as e:
                print(e)
                print(key)
            D[key][key_hps] = truths
            f3_displayed = False
            
            
            
            for x,y in truths.items():
            
                if not y:
                        
                    if not f2_displayed:
                        f2_displayed = True
                        print("\t" + key)
                    if not f3_displayed:
                        f3_displayed = True
                        print("\t\t" + key_hps)
                    print("\t\t\t"+x)
                
            if not truths:
                if not f2_displayed:
                    f2_displayed = True
                    print("\t" + key)
                if not f3_displayed:
                    f3_displayed = True
                    print("\t\t" + key_hps,end="\t")
                print("EMPTY")
            elif False not in truths.values():
                if not f2_displayed:
                    f2_displayed = True
                    print("\t" + key)
                if not f3_displayed:
                    f3_displayed = True
                    print("\t\t" + key_hps,end="\t")
                print("COMPLETE")
                
base = "src/"
basefolders = ["SB3", "CleanRL","TorchRL"]  
for _alg in ["sac"]:
    for b in basefolders:
        getExperiments(b,_alg)

