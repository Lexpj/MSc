import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
from scipy.stats import norm

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from os import listdir
from os.path import isfile, join
import pickle


def getCI(data,confidence=95):
    "data = list[tuple(mu, sigma)]"
    means = [m for m, s in data]
    n = 5

    # Calculate the z-score for the given confidence level
    alpha = 1 - confidence / 100
    z = norm.ppf(1 - alpha / 2)

    lower = [m - z * (s/np.sqrt(n)) for m, s in data]
    upper = [m + z * (s/np.sqrt(n)) for m, s in data]
    
    return means,lower,upper
    
def mean_std(d, window):
    try:
        return subsample(d, window)
    except:
        return np.array([]), np.array([]), np.array([])
        
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

    
def readFile(logfile, scalar):
    event_acc = EventAccumulator(logfile)
    event_acc.Reload()
    print("Available scalar tags:", event_acc.Tags()['scalars'])  # List all scalar tags
    # Extract the given scalar info
    try:
        scalar_events = event_acc.Scalars(scalar)
    except Exception as e:
        print("Something went wrong reading",logfile,":",e)
        return [], []
    steps = [e.step for e in scalar_events]
    values = [e.value for e in scalar_events]
    
    return steps, values
    
    
def extractCurve(path, scalar, window):

    # Make collection arrays for each path and repetition
    KV = {}
    
    # Test if cache exists, else
    try: 
        with open("./figs/cache/"+(path+scalar).replace("/","")[1:]+".pkl", "rb") as f:
            KV = pickle.load(f)          
            print("Using cached results of",path)
    except:        
        # Check out what is included in the folder
        folders = [x[0]+"/" for x in os.walk(path)][1:]

        # Extract the event loggers from each folder
        for rep in folders:
            log_file = [f for f in listdir(rep) if isfile(join(rep, f))]
            log_file = [rep+f for f in log_file if f[-3:] not in [".pt", "zip"]]
            steps = []
            values = []
        
            # Read out files and save cache
            
            for logfile in sorted(log_file):
            
                # Read the file, from cache if possible
                steps_, values_ = readFile(logfile, scalar)
                steps = steps + steps_
                values = values + values_
            
            steps = np.array(steps)
            values = np.array(values)
         
            # Save them for this experiment
            for key,value in zip(steps,values):
                KV[key] = KV.get(key,[]) + [value]
                
        if KV: # If no run was found, dont make a cache
            with open("./figs/cache/"+(path+scalar).replace("/","")[1:]+".pkl", "wb") as f:
                pickle.dump(KV, f)
                print("New cached results of",path)
        else:
            print("Cannot find results of",path)
           
    #print(KV)
    
    
    step,avg,std = mean_std(KV,window=window)

    return step, avg, std


